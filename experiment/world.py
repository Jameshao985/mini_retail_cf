"""Executable retail environment, deterministic user and evidence-based verifier."""
from __future__ import annotations

import copy
import json

from .tasks import action_name, decision, success_outcome


SYSTEM = '''你是订单助手，每次只输出一个JSON动作，不输出解释或Markdown。
动作格式（占位符必须替换为实际值）：
{"action":"get_order","order_id":"订单号"}
{"action":"cancel_order","order_id":"订单号"}
{"action":"refund_order","order_id":"订单号"}
{"action":"ask_user","field":"order_id"}
{"action":"respond","order_id":"订单号","outcome":"cancelled或refunded或refused"}
通用规则：缺少订单号必须ask_user，不能猜测。用户最新指定的订单号优先。执行写操作前必须成功get_order查询同一订单。
取消规则：pending/processing允许cancel_order；shipped/delivered禁止取消。
退货规则：只有status=delivered、used=false并且delivered_days不超过7天时，才允许refund_order；否则拒绝。
用户更正目标后必须查询新目标，不得操作旧目标。写操作成功后respond对应的cancelled或refunded。
条件不允许时直接respond outcome=refused。不得以工具拒绝代替自己的条件判断。'''


def parse(raw):
    try:
        action = json.loads(raw.strip())
        if not isinstance(action, dict):
            raise ValueError('Expected a JSON object')
        name = action.get('action')
        fields = {
            'get_order': {'action', 'order_id'},
            'cancel_order': {'action', 'order_id'},
            'refund_order': {'action', 'order_id'},
            'ask_user': {'action', 'field'},
            'respond': {'action', 'order_id', 'outcome'},
        }
        if name not in fields or set(action) != fields[name]:
            raise ValueError('Wrong action or schema')
        if name == 'ask_user':
            if action['field'] != 'order_id':
                raise ValueError('Unknown field')
        elif not isinstance(action['order_id'], str) or not action['order_id']:
            raise ValueError('order_id must be a nonempty string')
        if name == 'respond' and action['outcome'] not in ('cancelled', 'refunded', 'refused'):
            raise ValueError('Invalid outcome')
        return action, None
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return None, str(exc)


class World:
    def __init__(self, task):
        self.task = task
        self.state = copy.deepcopy(task['initial_state'])
        self.known_target = task['initial_target']
        self.followup_sent = False
        self.checked = set()
        self.executed = {}
        self.messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task['request']}]
        self.steps, self.errors = [], []
        self.done = False
        self.outcome = None

    def phase(self):
        if self.known_target is None:
            return 'missing_information'
        if self.task['followup'] and self.followup_sent:
            return 'after_user_update'
        return 'before_query' if self.known_target not in self.checked else 'after_query'

    def _allowed(self, target):
        order = self.state['orders'][target]
        if self.task['operation'] == 'cancel':
            return order['status'] in ('pending', 'processing')
        return order['status'] == 'delivered' and not order['used'] and order['delivered_days'] <= 7

    def expected(self):
        """The rule teacher uses only the visible target and successfully queried state."""
        target = self.known_target
        if target is None:
            return {'action': 'ask_user', 'field': 'order_id'}
        if target not in self.checked:
            return {'action': 'get_order', 'order_id': target}
        if target in self.executed:
            return {'action': 'respond', 'order_id': target, 'outcome': self.executed[target]}
        if self._allowed(target):
            return {'action': action_name(self.task), 'order_id': target}
        return {'action': 'respond', 'order_id': target, 'outcome': 'refused'}

    def apply(self, raw):
        action, error = parse(raw)
        step = dict(step=len(self.steps) + 1, messages=copy.deepcopy(self.messages), raw=raw,
                    action=action, expected=self.expected(), phase=self.phase(),
                    before=copy.deepcopy(self.state), observation=None, error=None)
        codes, observation, continuation = [], None, None
        if error:
            codes.append('protocol_error')
            self.done = True
        else:
            name = action['action']
            oid = action.get('order_id')
            if name == 'ask_user':
                if self.known_target is not None:
                    codes.append('unnecessary_clarification')
                    observation = {'error': 'ORDER_ID_ALREADY_PROVIDED'}
                else:
                    self.known_target = self.task['target']
                    continuation = self.task['clarification_reply']
            else:
                if self.known_target is None:
                    codes.append('missing_clarification')
                elif oid != self.known_target:
                    codes.append('stale_target' if self.followup_sent else 'wrong_target')
                order = self.state['orders'].get(oid)
                if name in ('get_order', 'cancel_order', 'refund_order') and order is None:
                    observation = {'success': False, 'error': 'ORDER_NOT_FOUND'}
                    codes.append('unknown_order')
                elif name == 'get_order':
                    self.checked.add(oid)
                    observation = dict(success=True, order_id=oid, **order)
                    if self.task['followup'] and not self.followup_sent:
                        self.followup_sent = True
                        self.known_target = self.task['target']
                        continuation = self.task['followup']
                elif name in ('cancel_order', 'refund_order'):
                    if name != action_name(self.task):
                        codes.append('wrong_operation')
                        observation = {'success': False, 'order_id': oid, 'error': 'WRONG_OPERATION'}
                    else:
                        if oid not in self.checked:
                            codes.append('missing_state_check')
                        if not self._allowed(oid):
                            codes.append('forbidden_action')
                            observation = {'success': False, 'order_id': oid, 'error': 'POLICY_FORBIDS_OPERATION'}
                        else:
                            outcome = success_outcome(self.task)
                            order['status'] = outcome
                            self.executed[oid] = outcome
                            observation = {'success': True, 'order_id': oid, 'status': outcome}
                elif name == 'respond':
                    self.done = True
                    self.outcome = action
                    if oid not in self.checked:
                        codes.append('missing_state_check')
                    if self.task['followup'] and not self.followup_sent:
                        codes.append('premature_response')
                    if action['outcome'] in ('cancelled', 'refunded') and self.executed.get(oid) != action['outcome']:
                        codes.append('false_success')
                    if action['outcome'] == 'refused' and decision(self.task) != 'refused':
                        codes.append('over_refusal')
        self.messages.append({'role': 'assistant', 'content': raw})
        if observation is not None:
            self.messages.append({'role': 'user', 'content': '工具返回：' + json.dumps(observation, ensure_ascii=False)})
        if continuation:
            self.messages.append({'role': 'user', 'content': continuation})
        step.update(after=copy.deepcopy(self.state), observation=observation,
                    user_continuation=continuation, error=error, violations=codes)
        self.steps.append(step)
        self.errors.extend(codes)

    def result(self):
        violations = list(self.errors)
        target = self.task['target']
        gold = decision(self.task)
        expected_status = (gold if gold != 'refused'
                           else self.task['initial_state']['orders'][target]['status'])
        terminal = self.state['orders'][target]['status'] == expected_status
        others_unchanged = all(order == self.task['initial_state']['orders'][oid]
                               for oid, order in self.state['orders'].items() if oid != target)
        if not terminal or not others_unchanged:
            violations.append('wrong_terminal_state')
        response_ok = self.outcome == {'action': 'respond', 'order_id': target, 'outcome': gold}
        if not response_ok:
            violations.append('wrong_response')
        if not self.done:
            violations.append('step_limit')
        return dict(task_id=self.task['task_id'], pair_id=self.task['pair_id'],
                    capability=self.task['capability'], operation=self.task['operation'],
                    pair_kind=self.task.get('pair_kind', 'boundary'), gold_outcome=gold,
                    task=copy.deepcopy(self.task), steps=self.steps, final_state=self.state,
                    outcome=self.outcome, violations=sorted(set(violations)),
                    task_success=not violations, terminal_correct=terminal and others_unchanged)


def rollout(task, policy, max_steps=7):
    env = World(task)
    for _ in range(max_steps):
        raw = policy(env.messages) if policy is not None else json.dumps(env.expected(), ensure_ascii=False)
        env.apply(raw)
        if env.done:
            break
    return env.result()


def teacher_examples(task):
    episode = rollout(task, None)
    if not episode['task_success']:
        raise ValueError(f'Rule teacher failed verification: {task["task_id"]}')
    return [dict(task_id=task['task_id'], group_id=task['group_id'], pair_id=task['pair_id'],
                 capability=task['capability'], messages=s['messages'], response=s['raw'],
                 teacher='rule_oracle', verified=True) for s in episode['steps']]
