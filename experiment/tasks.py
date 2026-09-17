"""Deterministic, disjoint retail tasks and minimal counterfactual pairs."""
from __future__ import annotations

import copy
import random

from .io import digest


# Fixed generator families. Clustering never sees these labels as text, but the
# experiment uses them to route an observed weakness to a suitable generator.
CAPABILITIES = (
    'precondition',
    'refund_window',
    'refund_condition',
    'clarification',
    'correction',
)

CANCEL_REQUESTS = {
    'discovery': ('请替我取消订单{a}。', '订单{a}我不要了，请办理取消。'),
    'train': ('麻烦取消编号为{a}的订单。', '请帮我撤销订单{a}。', '订单{a}不需要了，请实际取消。'),
    'dev': ('请执行订单{a}的取消操作。', '帮我终止订单{a}。'),
    'test': ('不需要这笔订单了，请取消{a}。', '请把订单{a}撤销掉。', '我改变主意了，取消{a}。'),
    'ood': ('撤销购买：订单编号{a}，请实际处理。', '终止编号{a}的交易并反馈结果。'),
}
REFUND_REQUESTS = {
    'discovery': ('请为订单{a}办理退货退款。', '我想退掉订单{a}的商品。'),
    'train': ('麻烦处理订单{a}的退货退款。', '订单{a}的商品我要退掉。', '请实际为{a}办理退货。'),
    'dev': ('请执行订单{a}的退货操作。', '帮我处理{a}的商品退回。'),
    'test': ('订单{a}的商品不合适，请办理退货。', '我想把{a}的商品退掉。', '请处理订单{a}的退货退款。'),
    'ood': ('发起售后：将交易{a}中的商品退回。', '请为编号{a}的已收货商品办理退回。'),
}
MISSING_REQUESTS = {
    'cancel': {
        'discovery': ('我想取消一笔订单。', '有个订单不需要了。'),
        'train': ('麻烦帮我取消一笔订单。', '请帮我撤销刚才的订单。'),
        'dev': ('请处理我的订单取消请求。', '我需要终止一笔订单。'),
        'test': ('有笔订单我不想要了，请帮我取消。', '请替我撤销那笔订单。'),
        'ood': ('我要撤销一笔购买，请实际处理。', '请终止我之前的一笔交易。'),
    },
    'refund': {
        'discovery': ('我想退掉一件商品。', '请帮我办理一笔退货。'),
        'train': ('麻烦帮我退掉一件商品。', '有笔订单需要退货退款。'),
        'dev': ('请处理我的商品退回请求。', '我需要办理一笔退货。'),
        'test': ('有件商品不合适，我想退货。', '请替我办理那件商品的退回。'),
        'ood': ('我要发起一笔售后退回。', '请处理我之前交易的商品退回。'),
    },
}
CORRECTIONS = {
    'discovery': ('更正一下，处理的是订单{target}。', '我刚说错了，目标是{target}。'),
    'train': ('刚才的编号需要确认，请以订单{target}为准。', '修改一下，请处理{target}。'),
    'dev': ('更新目标，实际需要处理订单{target}。', '请改为处理{target}。'),
    'test': ('请注意，我现在指定要处理的是{target}。', '最后确认，应当处理订单{target}。'),
    'ood': ('最终指令：处理{target}，以此条消息为最终依据。', '覆盖之前的编号，目标改成{target}。'),
}
EXTRAS = {
    'discovery': ('谢谢。', '请按规则办理。', ''),
    'train': ('麻烦了。', '请给出处理结果。', '辛苦帮忙处理。', ''),
    'dev': ('处理完告诉我。', '请核对条件。', ''),
    'test': ('请反馈办理情况。', '请先检查是否符合要求。', ''),
    'ood': ('请完成系统操作并反馈。', '请核实后再操作。', ''),
}
ITEMS = {
    'discovery': ['水杯', '键盘', '鼠标'],
    'train': ['耳机', '背包', '台灯', '充电器', '路由器', '外套'],
    'dev': ['手套', '书架', '帽子'],
    'test': ['雨伞', '风扇', '运动鞋', '咖啡机'],
    'ood': ['望远镜', '睡袋', '登山杖'],
}


def operation_for(capability, index):
    if capability == 'precondition':
        return 'cancel'
    if capability in ('refund_window', 'refund_condition'):
        return 'refund'
    return 'cancel' if index % 2 == 0 else 'refund'


def action_name(task):
    return 'cancel_order' if task['operation'] == 'cancel' else 'refund_order'


def success_outcome(task):
    return 'cancelled' if task['operation'] == 'cancel' else 'refunded'


def decision(task):
    order = task['initial_state']['orders'][task['target']]
    if task['operation'] == 'cancel':
        allowed = order['status'] in ('pending', 'processing')
    else:
        allowed = order['status'] == 'delivered' and not order['used'] and order['delivered_days'] <= 7
    return success_outcome(task) if allowed else 'refused'


def make_pair(split, capability, index, seed=42, namespace='default'):
    if split not in CANCEL_REQUESTS or capability not in CAPABILITIES:
        raise ValueError('Unknown split or capability')
    base_id = f'{namespace}/{split}/{capability}/{index}'
    rng = random.Random(f'{seed}:{base_id}')
    number = int(digest([seed, base_id])[:12], 16) % 90000000 + 10000000
    a, b = str(number), str(number + 1)
    item = rng.choice(ITEMS[split])
    price = rng.randint(20, 999)
    operation = operation_for(capability, index)
    requests = CANCEL_REQUESTS if operation == 'cancel' else REFUND_REQUESTS
    request = rng.choice(requests[split]).format(a=a) + rng.choice(EXTRAS[split])
    orders = {
        a: {'status': 'pending' if operation == 'cancel' else 'delivered', 'item': item,
            'price': price, 'used': False, 'delivered_days': 0 if operation == 'cancel' else rng.randint(1, 6)},
        b: {'status': 'pending' if operation == 'cancel' else 'delivered', 'item': item,
            'price': price, 'used': False, 'delivered_days': 0 if operation == 'cancel' else rng.randint(1, 6)},
    }
    task = dict(task_id=base_id + '/a', group_id=base_id, pair_id=base_id, split=split,
                capability=capability, operation=operation, initial_state={'orders': orders},
                request=request, initial_target=a, target=a, followup=None,
                clarification_reply=f'订单号是{a}。', intervention_variable='', variant=0)
    left, right = copy.deepcopy(task), copy.deepcopy(task)
    right.update(task_id=base_id + '/b', variant=1)
    if capability == 'precondition':
        left['initial_state']['orders'][a]['status'] = rng.choice(['pending', 'processing'])
        right['initial_state']['orders'][a]['status'] = rng.choice(['shipped', 'delivered'])
        variable = f'initial_state.orders.{a}.status'
    elif capability == 'refund_window':
        left['initial_state']['orders'][a]['delivered_days'] = rng.choice([3, 6, 7])
        right['initial_state']['orders'][a]['delivered_days'] = rng.choice([8, 10, 14])
        variable = f'initial_state.orders.{a}.delivered_days'
    elif capability == 'refund_condition':
        left['initial_state']['orders'][a]['used'] = False
        right['initial_state']['orders'][a]['used'] = True
        variable = f'initial_state.orders.{a}.used'
    elif capability == 'clarification':
        missing = rng.choice(MISSING_REQUESTS[operation][split])
        suffix = rng.choice(EXTRAS[split])
        left['request'] = missing + f'订单号是{a}。' + suffix
        right.update(request=missing + suffix, initial_target=None)
        variable = 'request.order_id_presence'
    else:
        followup = rng.choice(CORRECTIONS[split])
        left['followup'] = followup.format(target=a)
        right.update(followup=followup.format(target=b), target=b)
        variable = 'followup.target_order_id'
    for t in (left, right):
        t['intervention_variable'] = variable
    validate_pair(left, right)
    return [left, right]


def _diff(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for k in sorted(set(a) | set(b)):
            p = f'{path}.{k}' if path else k
            result += [p] if k not in a or k not in b else _diff(a[k], b[k], p)
        return result
    return [] if a == b else [path]


def validate_pair(a, b):
    for key in ('pair_id', 'group_id', 'capability', 'split', 'operation', 'intervention_variable'):
        if a[key] != b[key]:
            raise ValueError(f'Pair control differs: {key}')
    ignore = {'task_id', 'variant', 'pair_kind'}
    diffs = set(_diff({k: v for k, v in a.items() if k not in ignore},
                      {k: v for k, v in b.items() if k not in ignore}))
    capability = a['capability']
    expected = {
        'precondition': {a['intervention_variable']},
        'refund_window': {a['intervention_variable']},
        'refund_condition': {a['intervention_variable']},
        'clarification': {'request', 'initial_target'},
        'correction': {'followup', 'target'},
    }[capability]
    if diffs != expected:
        raise ValueError(f'Impure pair: {diffs}, expected {expected}')
    if capability in ('precondition', 'refund_window', 'refund_condition') and decision(a) == decision(b):
        raise ValueError('Boundary pair must change the correct outcome')
    if capability == 'clarification' and not (a['initial_target'] and b['initial_target'] is None):
        raise ValueError('Clarification pair must change information availability')
    return True


def make_split(split, pairs_per_capability, seed, namespace='eval'):
    tasks = [t for c in CAPABILITIES for i in range(pairs_per_capability)
             for t in make_pair(split, c, i, seed, namespace)]
    random.Random(seed).shuffle(tasks)
    return tasks


def make_protocol_tasks(count, seed, namespace='common_warmup'):
    """Shared allowed-side demonstrations for JSON/tool protocol before diagnosis."""
    families = ('precondition', 'refund_window', 'refund_condition')
    tasks = []
    for i in range(count):
        task = make_pair('train', families[i % len(families)], i, seed, namespace)[0]
        task['pair_id'] = None
        task['pair_kind'] = 'protocol_warmup'
        tasks.append(task)
    return tasks


def make_invariance(pairs_per_capability, seed):
    result = []
    for c in CAPABILITIES:
        for i in range(pairs_per_capability):
            a = make_pair('test', c, i, seed, 'invariance')[i % 2]
            b = copy.deepcopy(a)
            a['task_id'], b['task_id'] = a['group_id'] + '/original', a['group_id'] + '/price'
            b['initial_state']['orders'][b['target']]['price'] += 17
            for t in (a, b):
                t['pair_kind'] = 'invariance'
            result += [a, b]
    return result


def make_composition(pairs, seed):
    """Evaluation pairs where another refund condition masks the edited one.

    Both sides must be refused. This checks whether the student applies the full
    conjunction instead of treating one favourable field as sufficient.
    """
    result = []
    for i in range(pairs):
        family = 'refund_window' if i % 2 == 0 else 'refund_condition'
        pair = make_pair('ood', family, i, seed, 'composition')
        if family == 'refund_window':
            for task in pair:
                task['initial_state']['orders'][task['target']]['used'] = True
        else:
            for task in pair:
                task['initial_state']['orders'][task['target']]['delivered_days'] = 10
        if decision(pair[0]) != 'refused' or decision(pair[1]) != 'refused':
            raise ValueError('Composition control must mask both sides')
        for task in pair:
            task['capability'] = 'refund_composition'
            task['pair_kind'] = 'masked_boundary'
        result.extend(pair)
    return result


def assert_disjoint(*splits):
    groups, identifiers, payloads = set(), set(), set()
    for tasks in splits:
        g = {t['group_id'] for t in tasks}
        ids = {oid for t in tasks for oid in t['initial_state']['orders']}
        p = {digest([t['request'], t['initial_state'], t['followup']]) for t in tasks}
        if groups & g or identifiers & ids or payloads & p:
            raise ValueError('Split leakage: shared group, order ID or task payload')
        groups |= g
        identifiers |= ids
        payloads |= p
