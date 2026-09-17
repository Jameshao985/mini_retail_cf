"""Offline teacher tests. Test transports are fixtures, not real DeepSeek results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiment.io import read
from experiment.tasks import CAPABILITIES, make_pair
from experiment.teacher import DeepSeekTeacher, replay_candidates
from experiment.world import rollout


def control_payload(tasks):
    return {'trajectories': [{'task_id': t['task_id'], 'actions': [s['action'] for s in rollout(t, None)['steps']]}
                             for t in tasks]}


class TeacherTests(unittest.TestCase):
    def test_teacher_replay_all_capabilities(self):
        for capability in CAPABILITIES:
            tasks = make_pair('train', capability, 0)
            rows, episodes = replay_candidates(tasks, control_payload(tasks), 'unit_test_teacher')
            self.assertTrue(all(e['task_success'] for e in episodes))
            self.assertTrue(all(r['verified'] for r in rows))

    def test_invalid_side_rejects_whole_pair(self):
        tasks = make_pair('train', 'precondition', 0)
        payload = control_payload(tasks)
        payload['trajectories'][1]['actions'].insert(1, {'action': 'cancel_order', 'order_id': tasks[1]['target']})
        with self.assertRaises(ValueError):
            replay_candidates(tasks, payload, 'test')

    def test_duplicate_or_omitted_task_rejected(self):
        tasks = make_pair('train', 'precondition', 0)
        payload = control_payload(tasks)
        payload['trajectories'][1] = copy.deepcopy(payload['trajectories'][0])
        with self.assertRaises(ValueError):
            replay_candidates(tasks, payload, 'test')

    def test_cache_replays_without_rebilling_and_has_provenance(self):
        tasks = make_pair('train', 'correction', 0)
        calls = []
        def transport(body):
            calls.append(body)
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(control_payload(tasks))}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 20}}
        with tempfile.TemporaryDirectory() as folder:
            teacher = DeepSeekTeacher(folder, transport=transport)
            first = teacher.generate(tasks, [], 'targeted_cf')
            second = teacher.generate(tasks, [], 'targeted_cf')
            self.assertEqual(len(calls), 1)
            self.assertEqual(first[0], second[0])
            self.assertTrue(second[2]['cached'])
            self.assertEqual(read(Path(folder) / 'ledger.json')['attempts'][0]['status'], 'accepted')
            self.assertNotIn('Authorization', json.dumps(calls))

    def test_no_auto_retry_after_uncertain_request(self):
        tasks = make_pair('train', 'clarification', 0)
        calls = []
        def transport(body):
            calls.append(body)
            raise RuntimeError('Test network failure')
        with tempfile.TemporaryDirectory() as folder:
            teacher = DeepSeekTeacher(folder, transport=transport)
            with self.assertRaises(RuntimeError):
                teacher.generate(tasks, [], 'targeted_cf')
            with self.assertRaisesRegex(RuntimeError, 'previously attempted'):
                teacher.generate(tasks, [], 'targeted_cf')
            self.assertEqual(len(calls), 1)

    def test_call_cap_checked_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            teacher = DeepSeekTeacher(folder, max_calls=0,
                                      transport=lambda _: self.fail('Must not contact transport'))
            with self.assertRaisesRegex(RuntimeError, 'cap reached'):
                teacher.generate(make_pair('train', 'precondition', 0), [], 'targeted_cf')

    def test_truncated_response_never_accepted(self):
        tasks = make_pair('train', 'precondition', 0)
        with tempfile.TemporaryDirectory() as folder:
            teacher = DeepSeekTeacher(folder, transport=lambda _: {'choices': [{'finish_reason': 'length'}]})
            with self.assertRaisesRegex(ValueError, 'finish normally'):
                teacher.generate(tasks, [], 'targeted_cf')
            ledger = read(Path(folder) / 'ledger.json')
            self.assertEqual(ledger['attempts'][0]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
