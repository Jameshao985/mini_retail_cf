"""One offline Qwen implementation shared by rollout, embeddings and LoRA SFT."""
from __future__ import annotations

import gc
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .io import dump, jsonl


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prompt_ids(tokenizer, messages):
    # Rendering to text avoids version-specific BatchEncoding return types.
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    return tokenizer.encode(text, add_special_tokens=False)


class Student:
    def __init__(self, path, device='auto', seed=42, max_new_tokens=96, adapter=None):
        seed_all(seed)
        self.path = str(path)
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Check the CUDA PyTorch installation or use --device cpu.')
        self.dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.device == 'cuda' else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False,
            dtype=self.dtype, attn_implementation='sdpa').to(self.device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=False,
                                                  local_files_only=True)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def prompt_ids(self, messages):
        return prompt_ids(self.tokenizer, messages)

    def __call__(self, messages):
        ids = torch.tensor([self.prompt_ids(messages)], device=self.device)
        with torch.inference_mode():
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         do_sample=False, max_new_tokens=self.max_new_tokens,
                         pad_token_id=self.tokenizer.pad_token_id,
                         eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
        return self.tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def embed(self, texts):
        vectors = []
        # Backbone only: avoid allocating vocabulary logits for embedding extraction.
        core = self.model.get_base_model() if hasattr(self.model, 'peft_config') else self.model
        for text in texts:
            batch = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=192).to(self.device)
            with torch.inference_mode():
                hidden = core.model(**batch, use_cache=False).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1)
                v = (hidden.float() * mask).sum(1) / mask.sum(1)
                v = torch.nn.functional.normalize(v, dim=-1)
            vectors.append(v[0].cpu().numpy())
        return np.stack(vectors)

    def close(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def encode_example(tokenizer, example, max_length):
    prefix = prompt_ids(tokenizer, example['messages'])
    answer = tokenizer.encode(example['response'], add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prefix) + len(answer) > max_length:
        raise ValueError(f'SFT sample is {len(prefix) + len(answer)} tokens, exceeds max_length={max_length}; '
                         'increase --max-length. Samples are never silently truncated.')
    return {'input_ids': prefix + answer, 'labels': [-100] * len(prefix) + answer,
            'target_tokens': len(answer), 'task_id': example['task_id']}


def train_lora(model_path, rows, output, config, previous_adapter=None):
    from peft import LoraConfig, PeftModel, get_peft_model
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    student = Student(model_path, config['device'], config['seed'], config['max_new_tokens'])
    try:
        if previous_adapter:
            student.model = PeftModel.from_pretrained(student.model, str(previous_adapter), is_trainable=True,
                                                     local_files_only=True)
        else:
            student.model = get_peft_model(student.model, LoraConfig(
                task_type='CAUSAL_LM', r=config['lora_r'], lora_alpha=config['lora_r'] * 2,
                lora_dropout=0.05, bias='none', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
        model = student.model
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        model.train()
        encoded = [encode_example(student.tokenizer, row, config['max_length']) for row in rows]
        if not encoded:
            raise ValueError('Empty SFT dataset')
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'], weight_decay=0.01)
        # PEFT adapter parameters are fp32. GradScaler is needed only for fp16 CUDA.
        scaler = torch.amp.GradScaler('cuda', enabled=student.device == 'cuda' and student.dtype == torch.float16)
        rng = random.Random(config['seed'])
        order, cursor = [], 0
        logs = []
        target_tokens, input_tokens = 0, 0
        start = time.monotonic()
        for update in range(config['train_steps']):
            optimizer.zero_grad(set_to_none=True)
            samples = []
            for _ in range(config['grad_accum']):
                if cursor == len(order):
                    order = list(range(len(encoded)))
                    rng.shuffle(order)
                    cursor = 0
                samples.append(encoded[order[cursor]])
                cursor += 1
            # Weight by supervised tokens within an optimizer update, including partial lengths.
            denom = sum(s['target_tokens'] for s in samples)
            loss_value = 0.0
            for sample in samples:
                ids = torch.tensor([sample['input_ids']], device=student.device)
                labels = torch.tensor([sample['labels']], device=student.device)
                with torch.autocast(device_type=student.device, dtype=student.dtype,
                                    enabled=student.device == 'cuda'):
                    loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
                    weighted = loss * (sample['target_tokens'] / denom)
                if not torch.isfinite(weighted):
                    raise RuntimeError('Non-finite loss: aborting this training run')
                scaler.scale(weighted).backward()
                loss_value += float(weighted.detach())
                target_tokens += sample['target_tokens']
                input_tokens += len(sample['input_ids'])
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError('Non-finite gradient: no result will be marked complete')
            # Fixed warmup and decay schedules are identical across arms.
            warmup = max(1, round(config['train_steps'] * 0.1))
            factor = min(1., (update + 1) / warmup) * max(0.1, 1 - update / config['train_steps'])
            for group in optimizer.param_groups:
                group['lr'] = config['learning_rate'] * factor
            scaler.step(optimizer)
            scaler.update()
            record = dict(update=update + 1, loss=loss_value, lr=optimizer.param_groups[0]['lr'],
                          supervised_tokens=target_tokens, input_tokens=input_tokens,
                          elapsed_seconds=time.monotonic() - start)
            logs.append(record)
            jsonl(output / 'training_log.jsonl', logs)
            if update == 0 or (update + 1) % 5 == 0 or update + 1 == config['train_steps']:
                print(f'  SFT {update + 1}/{config["train_steps"]}: loss={loss_value:.4f}', flush=True)
        model.save_pretrained(output / 'adapter')
        student.tokenizer.save_pretrained(output / 'adapter')
        summary = dict(optimizer_updates=config['train_steps'], examples=len(encoded),
                       dataset_input_tokens=sum(len(s['input_ids']) for s in encoded),
                       dataset_supervised_tokens=sum(s['target_tokens'] for s in encoded),
                       seen_input_tokens=input_tokens, seen_supervised_tokens=target_tokens,
                       trainable_parameters=sum(p.numel() for p in trainable),
                       initial_loss=logs[0]['loss'], final_loss=logs[-1]['loss'],
                       elapsed_seconds=time.monotonic() - start, previous_adapter=str(previous_adapter) if previous_adapter else None,
                       peak_cuda_allocated_mb=torch.cuda.max_memory_allocated() / 2**20 if student.device == 'cuda' else None)
        dump(output / 'training_summary.json', summary)
        return summary
    finally:
        # Remaining local tensor references are freed when this function returns.
        student.close()
