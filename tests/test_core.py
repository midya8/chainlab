import copy
import itertools
import json
import math
import unittest
from chainlab import fit,infer,viterbi,validate_model


def model():
    return {'version':'chainlab_v1','states':['A','B'],'symbols':['x','y'],
            'initial':[.6,.4],'transition':[[.7,.3],[.2,.8]],'emission':[[.8,.2],[.1,.9]]}


def enumerate_paths(m,sequence):
    obs=[m['symbols'].index(s) for s in sequence];paths=[]
    for path in itertools.product(range(len(m['states'])),repeat=len(obs)):
        p=m['initial'][path[0]]*m['emission'][path[0]][obs[0]]
        for t in range(1,len(obs)):p*=m['transition'][path[t-1]][path[t]]*m['emission'][path[t]][obs[t]]
        paths.append((path,p))
    return paths


class TestCore(unittest.TestCase):
    def test_forward_backward_against_exhaustive_all_marginals(self):
        m=model();seq=['x','y','x'];paths=enumerate_paths(m,seq);total=sum(p for _,p in paths);r=infer(m,seq)
        self.assertAlmostEqual(r['log_likelihood'],math.log(total),places=13)
        for t in range(len(seq)):
            for i in range(2):
                ref=sum(p for path,p in paths if path[t]==i)/total
                self.assertAlmostEqual(r['state_posteriors'][t][i],ref,places=13)
        for t in range(len(seq)-1):
            for i in range(2):
                for j in range(2):
                    ref=sum(p for path,p in paths if path[t]==i and path[t+1]==j)/total
                    self.assertAlmostEqual(r['transition_posteriors'][t][i][j],ref,places=13)

    def test_viterbi_against_enumeration_best_path_and_scores(self):
        m=model();seq=['x','y','x','y'];paths=enumerate_paths(m,seq);path,p=max(paths,key=lambda row:row[1]);total=sum(p for _,p in paths)
        r=viterbi(m,seq);self.assertEqual(r['state_indices'],list(path))
        self.assertAlmostEqual(r['log_joint_probability'],math.log(p),places=13)
        self.assertAlmostEqual(r['log_path_posterior'],math.log(p/total),places=13)
        self.assertLessEqual(r['log_joint_probability'],r['log_observation_likelihood'])

    def test_posterior_rows_and_pair_marginals_consistent(self):
        r=infer(model(),['x','y']*6)
        for row in r['state_posteriors']:self.assertAlmostEqual(sum(row),1,places=14)
        for t,pair in enumerate(r['transition_posteriors']):
            self.assertAlmostEqual(sum(map(sum,pair)),1,places=14)
            for i in range(2):
                self.assertAlmostEqual(sum(pair[i]),r['state_posteriors'][t][i],places=13)
                self.assertAlmostEqual(sum(row[i] for row in pair),r['state_posteriors'][t+1][i],places=13)

    def test_single_observation_has_no_transition_posteriors(self):
        r=infer(model(),['x']);self.assertEqual(r['transition_posteriors'],[])
        self.assertAlmostEqual(r['state_posteriors'][0][0],.48/.52)
        self.assertEqual(viterbi(model(),['x'])['path'],['A'])

    def test_log_space_rare_sequence_survives_probability_underflow(self):
        m={'version':'chainlab_v1','states':['only'],'symbols':['rare','common'],
           'initial':[1],'transition':[[1]],'emission':[[1e-6,1-1e-6]]}
        r=infer(m,['rare']*256);v=viterbi(m,['rare']*256)
        self.assertAlmostEqual(r['log_likelihood'],256*math.log(1e-6),places=9)
        self.assertEqual(math.exp(r['log_likelihood']),0)
        self.assertEqual(r['state_posteriors'],[[1.]]*256)
        self.assertAlmostEqual(v['log_path_posterior'],0)

    def test_impossible_observation_no_smoothing_fallback(self):
        m=model();m['emission']=[[1,0],[1,0]]
        for function in (infer,viterbi):
            with self.assertRaisesRegex(ValueError,'zero probability'):function(m,['y'])
        with self.assertRaises(ValueError):fit(m,[['y']])

    def test_marginal_mode_path_can_be_invalid_while_viterbi_is_valid(self):
        m={'version':'chainlab_v1','states':['A','B','C'],'symbols':['x'],'initial':[.4,.35,.25],
           'transition':[[0,.5,.5],[1,0,0],[1,0,0]],'emission':[[1],[1],[1]]}
        r=infer(m,['x','x']);v=viterbi(m,['x','x'])
        self.assertEqual(r['posterior_modes'],['A','A']);self.assertEqual(m['transition'][0][0],0)
        self.assertEqual(v['path'],['B','A']);self.assertAlmostEqual(math.exp(v['log_path_posterior']),.35)

    def test_local_exact_ties_keep_first_state_index(self):
        m=model();m.update(initial=[.5,.5],transition=[[.5,.5],[.5,.5]],emission=[[.5,.5],[.5,.5]])
        self.assertEqual(viterbi(m,['x']*4)['path'],['A']*4)
        self.assertEqual(infer(m,['x']*4)['posterior_modes'],['A']*4)

    def test_manual_em_update_against_enumerated_expected_counts(self):
        m=model();sequences=[['x','y','x'],['y','x']];n=2;start=[0.,0.];trans=[[0.]*n for _ in range(n)];emit=[[0.]*n for _ in range(n)]
        for seq in sequences:
            paths=enumerate_paths(m,seq);total=sum(p for _,p in paths)
            for path,p in paths:
                weight=p/total;start[path[0]]+=weight
                for t,state in enumerate(path):
                    emit[state][m['symbols'].index(seq[t])]+=weight
                    if t:trans[path[t-1]][state]+=weight
        r=fit(m,sequences,iterations=1)['model']
        for expected,actual in zip(start,r['initial']):self.assertAlmostEqual(expected/len(sequences),actual,places=13)
        for counts,key in ((trans,'transition'),(emit,'emission')):
            for i in range(n):
                for j in range(n):self.assertAlmostEqual(r[key][i][j],counts[i][j]/sum(counts[i]),places=13)

    def test_em_likelihood_monotonic_and_final_model_matches_history(self):
        seq=[['x','x','y','y','y'],['y','x','x','x'],['x','y','y']];r=fit(model(),seq,iterations=15,tolerance=0)
        losses=[h['log_likelihood'] for h in r['history']]
        self.assertTrue(all(b>=a for a,b in zip(losses,losses[1:])))
        self.assertGreater(losses[-1],losses[0])
        self.assertAlmostEqual(losses[-1],sum(infer(r['model'],s)['log_likelihood'] for s in seq),places=12)
        validate_model(r['model'])

    def test_sequence_boundaries_reset_and_no_cross_transitions(self):
        m=model();m.update(initial=[.5,.5],transition=[[1,0],[0,1]],emission=[[1,0],[0,1]])
        r=fit(m,[['x'],['y']],iterations=3)
        self.assertEqual(r['model']['initial'],[.5,.5]);self.assertEqual(r['model']['transition'],m['transition'])
        self.assertEqual(r['diagnostics']['within_sequence_transitions'],0)
        self.assertEqual(r['history'][1]['zero_occupancy_transition_rows_retained'],[0,1])
        with self.assertRaises(ValueError):infer(m,['x','y'])

    def test_zero_occupancy_rows_retained_not_replaced_by_uniform(self):
        m=model();m.update(initial=[1,0],transition=[[1,0],[0,1]])
        r=fit(m,[['x']*3],iterations=1)
        self.assertEqual(r['model']['emission'][1],m['emission'][1]);self.assertEqual(r['model']['transition'][1],[0,1])
        self.assertEqual(r['history'][1]['zero_occupancy_emission_rows_retained'],[1])
        self.assertEqual(r['history'][1]['zero_occupancy_transition_rows_retained'],[1])

    def test_unsmoothed_absent_symbol_can_lose_support(self):
        r=fit(model(),[['x']*4],iterations=1)
        self.assertTrue(all(row[1]==0 for row in r['model']['emission']))
        with self.assertRaises(ValueError):infer(r['model'],['y'])

    def test_single_state_training_is_empirical_symbol_frequency(self):
        m={'version':'chainlab_v1','states':['only'],'symbols':['x','y'],'initial':[1],
           'transition':[[1]],'emission':[[.5,.5]]}
        r=fit(m,[['x','x','y'],['x']],iterations=5)
        self.assertEqual(r['model']['emission'],[[.75,.25]])
        self.assertEqual(r['model']['initial'],[1]);self.assertEqual(r['model']['transition'],[[1]])
        self.assertEqual(r['stop_reason'],'improvement_tolerance')

    def test_no_input_mutation_and_serialized_model_roundtrip(self):
        m=model();seq=[['x','y'],['y','y']];before=copy.deepcopy((m,seq))
        r=fit(m,seq,iterations=2)
        self.assertEqual((m,seq),before)
        clone=json.loads(json.dumps(r['model']))
        self.assertEqual(infer(clone,seq[0]),infer(r['model'],seq[0]))
        self.assertEqual(viterbi(clone,seq[0]),viterbi(r['model'],seq[0]))

    def test_label_permutation_preserves_likelihood(self):
        a=model();b=copy.deepcopy(a);b['states'].reverse();b['initial'].reverse();b['emission'].reverse()
        b['transition']=[list(reversed(row)) for row in reversed(b['transition'])]
        x=infer(a,['x','y']);y=infer(b,['x','y'])
        self.assertAlmostEqual(x['log_likelihood'],y['log_likelihood'])
        for u,v in zip(x['state_posteriors'],y['state_posteriors']):
            self.assertAlmostEqual(u[0],v[1]);self.assertAlmostEqual(u[1],v[0])

    def test_strict_model_schema_and_probability_rows(self):
        variants=[]
        for key,value in [('version','other'),('states',['A','A']),('symbols',[]),('initial',[.6,.5]),('transition',[[1,0]]),('emission',[[1,0],[True,False]])]:
            m=model();m[key]=value;variants.append(m)
        m=model();m['extra']=1;variants.append(m)
        m=model();m['initial']=[math.nan,.4];variants.append(m)
        m=model();m['emission'][0]=[-.1,1.1];variants.append(m)
        for m in variants:
            with self.assertRaises(ValueError):validate_model(m)

    def test_sequence_and_training_bounds(self):
        for seq in ([],['unknown'],[0],['x']*257,'x'):
            with self.assertRaises(ValueError):infer(model(),seq)
        for seqs in ([],[['x']]*33,[['x']*256]*9,[[]]):
            with self.assertRaises(ValueError):fit(model(),seqs)
        for kwargs in ({'iterations':True},{'iterations':0},{'iterations':101},{'tolerance':-1},{'tolerance':math.inf},{'tolerance':True}):
            with self.assertRaises(ValueError):fit(model(),[['x']],**kwargs)

    def test_input_probability_rows_are_not_renormalized(self):
        m=model();m['initial']=[.6,.4+2e-13]
        self.assertEqual(validate_model(m)['initial'],m['initial'])

    def test_tiny_decreasing_em_proposal_is_rejected_without_history_mutation(self):
        from unittest.mock import patch
        from chainlab.core import _infer
        calls=[]
        def simulated(m,obs):
            result=_infer(m,obs);calls.append(1)
            if len(calls)==3:return (result[0]-1e-10,)+result[1:]
            return result
        m={'version':'chainlab_v1','states':['only'],'symbols':['x'],'initial':[1],'transition':[[1]],'emission':[[1]]}
        with patch('chainlab.core._infer',side_effect=simulated):r=fit(m,[['x']],iterations=2)
        self.assertEqual(r['stop_reason'],'tiny_numerical_decrease_rejected')
        self.assertEqual(len(r['history']),1);self.assertEqual(r['model'],m)
        self.assertLess(r['rejected_proposal']['change'],0)

    def test_material_decreasing_em_proposal_fails_loudly(self):
        from unittest.mock import patch
        from chainlab.core import _infer
        calls=[]
        def simulated(m,obs):
            result=_infer(m,obs);calls.append(1)
            if len(calls)==3:return (result[0]-.001,)+result[1:]
            return result
        m={'version':'chainlab_v1','states':['only'],'symbols':['x'],'initial':[1],'transition':[[1]],'emission':[[1]]}
        with patch('chainlab.core._infer',side_effect=simulated):
            with self.assertRaisesRegex(ValueError,'decreased'):fit(m,[['x']])
