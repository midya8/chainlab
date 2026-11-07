import copy
import io
import itertools
import json
import math
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
from chainlab import infer,filter_sequence,forecast
from chainlab.cli import main
from chainlab.demo import run_online_demo
from test_core import model,enumerate_paths


class TestOnline(unittest.TestCase):
    def test_filter_every_prefix_against_exact_enumeration(self):
        m=model();seq=['x','y','x','y'];r=filter_sequence(m,seq)
        for t in range(len(seq)):
            paths=enumerate_paths(m,seq[:t+1]);total=sum(p for _,p in paths)
            self.assertAlmostEqual(r['prefix_log_likelihoods'][t],math.log(total),places=13)
            for i in range(2):
                expected=sum(p for path,p in paths if path[-1]==i)/total
                self.assertAlmostEqual(r['filtered_state_posteriors'][t][i],expected,places=13)

    def test_predictive_scores_use_only_preceding_prefix_and_match_likelihood(self):
        m=model();seq=['x','y','x'];r=filter_sequence(m,seq)
        self.assertAlmostEqual(sum(r['observed_log_predictive']),infer(m,seq)['log_likelihood'],places=13)
        for t in range(len(seq)):
            self.assertAlmostEqual(sum(r['prior_predictive_symbol_probabilities'][t]),1)
            self.assertAlmostEqual(math.log(r['prior_predictive_symbol_probabilities'][t][m['symbols'].index(seq[t])]),r['observed_log_predictive'][t])
        self.assertAlmostEqual(r['prior_predictive_symbol_probabilities'][0][0],.6*.8+.4*.1)
        # A different observed suffix cannot alter the already-returned prefix rows.
        other=filter_sequence(m,['x','x','y'])
        self.assertEqual(r['filtered_state_posteriors'][0],other['filtered_state_posteriors'][0])
        self.assertEqual(r['prior_predictive_symbol_probabilities'][1],other['prior_predictive_symbol_probabilities'][1])

    def test_filter_is_not_full_sequence_smoothing(self):
        m={'version':'chainlab_v1','states':['A','B'],'symbols':['x','y'],
           'initial':[.5,.5],'transition':[[1,0],[0,1]],'emission':[[1,0],[.5,.5]]}
        f=filter_sequence(m,['x','y']);s=infer(m,['x','y'])
        self.assertAlmostEqual(f['filtered_state_posteriors'][0][0],2/3)
        self.assertEqual(s['state_posteriors'][0][0],0)
        self.assertEqual(f['filtered_state_posteriors'][-1],s['state_posteriors'][-1])

    def test_log_messages_rescue_a_raw_probability_underflow_path(self):
        tiny=1e-320
        m={'version':'chainlab_v1','states':['A','B'],'symbols':['common','rare'],
           'initial':[1.,tiny],'transition':[[1,0],[0,1]],'emission':[[1,0],[1.,1e-300]]}
        r=filter_sequence(m,['common','rare'])
        self.assertTrue(math.isfinite(r['log_likelihood']))
        self.assertAlmostEqual(r['log_likelihood'],math.log(tiny)+math.log(1e-300),places=10)
        self.assertEqual(r['prior_predictive_symbol_probabilities'][1][1],0)
        self.assertEqual(r['filtered_state_posteriors'][-1],[0.,1.])
        self.assertAlmostEqual(r['log_likelihood'],infer(m,['common','rare'])['log_likelihood'],places=10)

    def test_forecast_transitions_once_before_horizon_one_emission(self):
        m={'version':'chainlab_v1','states':['A','B'],'symbols':['x','y'],
           'initial':[1,0],'transition':[[0,1],[1,0]],'emission':[[1,0],[0,1]]}
        r=forecast(m,['x'],3)
        self.assertEqual([row['symbol_probabilities'] for row in r['forecasts']],[[0,1],[1,0],[0,1]])
        self.assertEqual([row['horizon'] for row in r['forecasts']],[1,2,3])

    def test_forecast_marginals_against_exact_future_path_sum(self):
        m=model();seq=['x','y'];observed_paths=enumerate_paths(m,seq);total=sum(p for _,p in observed_paths)
        r=forecast(m,seq,3)
        for h,row in enumerate(r['forecasts'],1):
            expected=[0.,0.]
            for past,p in observed_paths:
                for future in itertools.product(range(2),repeat=h):
                    weight=p/total;last=past[-1]
                    for state in future:weight*=m['transition'][last][state];last=state
                    expected[last]+=weight
            for state in range(2):self.assertAlmostEqual(row['state_probabilities'][state],expected[state],places=13)
            for symbol in range(2):
                self.assertAlmostEqual(row['symbol_probabilities'][symbol],sum(expected[i]*m['emission'][i][symbol] for i in range(2)),places=13)

    def test_forecast_marginals_are_not_independent_future_evidence(self):
        m={'version':'chainlab_v1','states':['A','B'],'symbols':['x','y'],
           'initial':[.5,.5],'transition':[[1,0],[0,1]],'emission':[[1,0],[.5,.5]]}
        r=forecast(m,['x'],2);one,two=[row['symbol_probabilities'][1] for row in r['forecasts']]
        joint=math.exp(infer(m,['x','y','y'])['log_likelihood']-infer(m,['x'])['log_likelihood'])
        self.assertAlmostEqual(one,1/6);self.assertAlmostEqual(two,1/6)
        self.assertAlmostEqual(joint,1/12);self.assertNotAlmostEqual(one*two,joint)

    def test_online_input_validation_impossible_prefix_and_no_mutation(self):
        m=model();seq=['x','y'];before=copy.deepcopy((m,seq))
        filter_sequence(m,seq);forecast(m,seq,32);self.assertEqual((m,seq),before)
        for h in (0,33,True,1.5):
            with self.assertRaises(ValueError):forecast(m,seq,h)
        for function in (filter_sequence,forecast):
            with self.assertRaises(ValueError):function(m,[])
            with self.assertRaises(ValueError):function(m,['unknown'])
            bad=copy.deepcopy(m);bad['emission']=[[1,0],[1,0]]
            with self.assertRaisesRegex(ValueError,'zero probability'):function(bad,['y'])

    def test_online_cli_and_wrong_schema_error_no_partial_json(self):
        for command,args,request,code in [('filter',[],{'model':model(),'sequence':['x','y']},0),
            ('forecast',['--horizon','2'],{'model':model(),'sequence':['x']},0),
            ('forecast',['--horizon','0'],{'model':model(),'sequence':['x']},2),
            ('filter',[],{'model':model(),'sequence':['x'],'extra':1},2)]:
            out,err=io.StringIO(),io.StringIO()
            with patch('sys.stdin',io.StringIO(json.dumps(request))),redirect_stdout(out),redirect_stderr(err):
                self.assertEqual(main([command,'--input','-']+args),code)
            if code==0:self.assertIsInstance(json.loads(out.getvalue()),dict)
            else:self.assertEqual(out.getvalue(),'');self.assertTrue(err.getvalue().startswith('error:'))

    def test_online_demo_is_measured_repeatable_and_prefix_only(self):
        r=run_online_demo();self.assertEqual(r,run_online_demo())
        audit=r['trained_fixture_likelihood_audit']
        self.assertEqual(audit['observations'],480);self.assertLess(audit['absolute_difference'],1e-10)
        self.assertAlmostEqual(audit['mean_negative_log_predictive'],.7756309218540562)
        ex=r['future_information_counterexample']
        self.assertAlmostEqual(ex['filtered_state_posteriors'][0][0],2/3)
        self.assertEqual(ex['full_sequence_smoothed_posteriors'][0][0],0)
        self.assertEqual(r['trained_prefix_example']['forecast']['conditioning_observations'],10)
