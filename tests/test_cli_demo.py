import io
import json
import math
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
from chainlab.cli import main
from chainlab.demo import run_demo
from test_core import model


class TestCLI(unittest.TestCase):
    def invoke(self,args,request):
        text=request if isinstance(request,str) else json.dumps(request)
        out,err=io.StringIO(),io.StringIO()
        with patch('sys.stdin',io.StringIO(text)),redirect_stdout(out),redirect_stderr(err):code=main(args)
        return code,out.getvalue(),err.getvalue()

    def test_fit_infer_viterbi_cli_artifact_roundtrip(self):
        request={'model':model(),'sequences':[['x','y','y'],['y','x','x']]}
        code,out,err=self.invoke(['fit','--input','-','--iterations','3'],request)
        self.assertEqual(code,0);self.assertEqual(err,'');trained=json.loads(out)['model']
        for command in ('infer','viterbi'):
            code,out,err=self.invoke([command,'--input','-'],{'model':trained,'sequence':['x','y']})
            self.assertEqual(code,0);self.assertEqual(err,'');self.assertIsInstance(json.loads(out),dict)

    def test_bad_json_schema_and_impossible_sequence_no_partial_output(self):
        m=model();m['emission']=[[1,0],[1,0]]
        for request in ('{bad',{}, {'model':m,'sequence':['y']},{'model':model(),'sequence':['x'],'test':[]}):
            code,out,err=self.invoke(['infer','--input','-'],request)
            self.assertEqual(code,2);self.assertEqual(out,'');self.assertTrue(err.startswith('error:'))

    def test_demo_repeatable_measured_controls_and_decoding_counterexample(self):
        r=run_demo();self.assertEqual(r,run_demo());arms=r['arms']
        self.assertAlmostEqual(arms['trained_hmm']['test_mean_negative_log_likelihood'],.7756309218540564)
        self.assertAlmostEqual(arms['training_iid_symbol_control']['test_mean_negative_log_likelihood'],.9859930334114989)
        self.assertAlmostEqual(arms['supplied_initialization']['test_mean_negative_log_likelihood'],1.017637767116141)
        self.assertAlmostEqual(arms['generating_model_reference']['test_mean_negative_log_likelihood'],.7716741801842274)
        self.assertEqual(r['training']['diagnostics']['within_sequence_transitions'],780)
        self.assertEqual(r['posterior_mode_counterexample']['posterior_modes'],['A','A'])
        self.assertEqual(r['posterior_mode_counterexample']['viterbi_path'],['B','A'])
        self.assertEqual(r['rare_symbol_sequence']['exp_log_likelihood'],0)
        self.assertTrue(math.isfinite(r['rare_symbol_sequence']['log_likelihood']))
        history=r['training']['history'];self.assertEqual(len(history),21)
        self.assertTrue(all(b['log_likelihood']>=a['log_likelihood'] for a,b in zip(history,history[1:])))
