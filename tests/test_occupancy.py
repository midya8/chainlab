"""Cartesian path sums independent of the visit-count message recurrence."""
import copy
import itertools
import json
import random
import unittest
from fractions import Fraction as F
from pathlib import Path
from chainlab.occupancy import visit_distribution
from chainlab.pathrisk import audit

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = [c for kind in ('core', 'rounded') for c in json.loads(
    (ROOT / ('chainlab/artifacts/path-controls-' + kind + '.json')).read_text())]


def oracle(payload, selected):
    m = payload['model']
    n = len(m['states'])
    obs = [m['symbols'].index(s) for s in payload['sequence']]
    joint = [F(0) for _ in range(len(obs) + 1)]
    for path in itertools.product(range(n), repeat=len(obs)):
        p = F(m['initial'][path[0]]) * F(m['emission'][path[0]][obs[0]])
        for t in range(1, len(obs)):
            p *= F(m['transition'][path[t-1]][path[t]]) * F(m['emission'][path[t]][obs[t]])
        joint[sum(m['states'][i] in selected for i in path)] += p
    return joint


class VisitTests(unittest.TestCase):
    def assert_law(self, inp, selected, cartesian=True):
        r = visit_distribution(inp, selected)
        probabilities = list(map(F, r['posterior_count_probabilities']))
        joint = list(map(F, r['joint_count_mass']))
        self.assertEqual(sum(joint), F(r['evidence']))
        self.assertEqual(probabilities, [v / sum(joint) for v in joint])
        self.assertEqual(sum(probabilities), 1)
        self.assertTrue(all(v >= 0 for v in probabilities))
        if cartesian:
            self.assertEqual(joint, oracle(inp, selected))
        for k, tail in enumerate(r['posterior_count_at_least']):
            self.assertEqual(F(tail), sum(probabilities[k:]))
        mean = sum(k*p for k, p in enumerate(probabilities))
        self.assertEqual(F(r['expected_count']), mean)
        self.assertEqual(F(r['count_variance']), sum(k*k*p for k, p in enumerate(probabilities))-mean*mean)
        # A second production API is only a consistency check, not the oracle.
        reference = audit(inp)
        indices = [inp['model']['states'].index(s) for s in selected]
        self.assertEqual(mean, sum(F(row[i]) for row in reference['state'] for i in indices))
        for t, layer in enumerate(r['prefix_joint_count_mass']):
            prefix = dict(model=inp['model'], sequence=inp['sequence'][:t+1])
            prefix_joint = [sum(F(row[k]) for row in layer) for k in range(t+2)]
            if cartesian:
                self.assertEqual(prefix_joint, oracle(prefix, selected))
            self.assertEqual(sum(prefix_joint), F(audit(prefix)['evidence']))
        return r

    def test_saved_controls_all_selections_and_exact_complements(self):
        for c in CONTROLS:
            inp = c['input']; names = inp['model']['states']; length = len(inp['sequence'])
            for mask in range(1 << len(names)):
                selected = [s for i, s in enumerate(names) if mask >> i & 1]
                r = self.assert_law(inp, selected, len(names)**length <= 5000)
                complement = visit_distribution(inp, [s for s in names if s not in selected])
                self.assertEqual(r['posterior_count_probabilities'], complement['posterior_count_probabilities'][::-1])
                self.assertEqual(r['count_variance'], complement['count_variance'])

    def test_120_random_models_every_prefix_cartesian_oracle(self):
        rng = random.Random(39127)
        def row(n):
            xs = [rng.randrange(5) for _ in range(n)]
            if not sum(xs): xs[0] = 1
            return [str(F(x, sum(xs))) for x in xs]
        for _ in range(120):
            n = rng.randrange(1, 4)
            m = dict(version='chainlab_rational_v1', states=['s'+str(i) for i in range(n)],
                     symbols=['x','y'], initial=row(n), transition=[row(n) for _ in range(n)],
                     emission=[row(2) for _ in range(n)])
            inp = dict(model=m, sequence=[rng.choice(m['symbols']) for _ in range(rng.randrange(1, 7))])
            selected = [s for s in m['states'] if rng.randrange(2)]
            if not sum(oracle(inp, selected)):
                with self.assertRaisesRegex(ValueError, 'zero probability'): visit_distribution(inp, selected)
            else: self.assert_law(inp, selected)

    def test_identical_marginals_do_not_determine_the_count_law(self):
        independent = next(c['input'] for c in CONTROLS if c['name']=='independent')
        locked = next(c['input'] for c in CONTROLS if c['name']=='locked')
        # Both controls have five positions with probability1/2 in each state.
        self.assertEqual(audit(independent)['state'], audit(locked)['state'])
        a = self.assert_law(independent, [independent['model']['states'][0]])
        b = self.assert_law(locked, [locked['model']['states'][0]])
        self.assertEqual(a['expected_count'], '5/2'); self.assertEqual(b['expected_count'], '5/2')
        self.assertEqual(a['posterior_count_probabilities'], ['1/32','5/32','5/16','5/16','5/32','1/32'])
        self.assertEqual(b['posterior_count_probabilities'], ['1/2','0','0','0','0','1/2'])
        self.assertEqual(a['count_variance'], '5/4'); self.assertEqual(b['count_variance'], '25/4')

    def test_single_position_empty_full_and_no_mutation(self):
        inp = copy.deepcopy(CONTROLS[0]['input']); inp['sequence'] = inp['sequence'][:1]
        before = copy.deepcopy(inp)
        for selected in ([], inp['model']['states'], inp['model']['states'][::-1]):
            original = selected[:]; r = self.assert_law(inp, selected)
            self.assertEqual(selected, original); self.assertEqual(inp, before)
            self.assertEqual(r['count_variance'], '0')
            self.assertEqual(r, json.loads(json.dumps(r, allow_nan=False)))
        self.assertEqual(visit_distribution(inp, [])['posterior_count_probabilities'], ['1','0'])
        self.assertEqual(visit_distribution(inp, inp['model']['states'])['posterior_count_probabilities'], ['0','1'])

    def test_state_permutation_and_selection_order_preserve_law(self):
        inp = copy.deepcopy(CONTROLS[-1]['input']); m = inp['model']; selected = m['states'][:1]
        a = self.assert_law(inp, selected, False)
        m['states'].reverse(); m['initial'].reverse(); m['emission'].reverse()
        m['transition'] = [row[::-1] for row in m['transition'][::-1]]
        b = self.assert_law(inp, selected, False)
        for key in ('evidence','joint_count_mass','posterior_count_probabilities','posterior_count_at_least','expected_count','count_variance'):
            self.assertEqual(a[key], b[key])
        self.assertEqual(visit_distribution(inp, m['states']), visit_distribution(inp, m['states'][::-1]))

    def test_invalid_selection_schema_impossibility_and_work_bound(self):
        inp = copy.deepcopy(CONTROLS[0]['input']); name = inp['model']['states'][0]
        for selected in (None, name, [name,name], ['unknown'], [True], [{}]):
            with self.assertRaises(ValueError): visit_distribution(inp, selected)
        bad = copy.deepcopy(inp); bad['unexpected'] = 1
        with self.assertRaises(ValueError): visit_distribution(bad, [])
        bad = copy.deepcopy(inp); bad['sequence'] *= 33
        with self.assertRaises(ValueError): visit_distribution(bad, [])
        bad = copy.deepcopy(inp); bad['model']['emission'] = [['0','1'] for _ in bad['model']['states']]
        bad['model']['symbols'] = ['x','y']; bad['sequence'] = ['x']
        with self.assertRaisesRegex(ValueError, 'zero probability'): visit_distribution(bad, [])
        # Coprime transition denominators can exceed the existing rational work cap.
        m = dict(version='chainlab_rational_v1', states=['a','b','c','d'], symbols=['x'],
                 initial=['1','0','0','0'], emission=[['1']]*4, transition=[])
        for d in (2**127-1,2**126-5,2**125-9,2**124-3):
            m['transition'].append([str(F(1,d)),str(F(2,d)),str(F(3,d)),str(F(d-6,d))])
        with self.assertRaisesRegex(ValueError,'8192-bit'):
            visit_distribution(dict(model=m,sequence=['x']*32), ['a'])
