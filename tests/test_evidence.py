import copy,itertools,json,math,subprocess,sys,tempfile,unittest
from pathlib import Path
from decimal import Decimal as D,localcontext
from chainlab.evidence import inspect
from chainlab.evidence_export import build,saved,assert_reproduces

class EvidenceTests(unittest.TestCase):
    def test_original_and_all_archives_reproduce(self):
        assert_reproduces(build(),saved())
        for bad in [1.00000001,float('nan'),True]:
            with self.assertRaises(AssertionError):assert_reproduces([bad],[1.])
    def test_all_edits_and_prefixes(self):
        r=saved();count=0;prefixes=0
        for a in r['arms'].values():
            for edit in a['edits']:
                seq=list(a['sequence']);seq[edit['position']]=edit['symbol'];expected=edit['result']
                if 'error' in expected:
                    with self.assertRaises(ValueError):inspect(a['model'],seq,a['baseline']['cut'],5)
                else:assert_reproduces(inspect(a['model'],seq,a['baseline']['cut'],5),expected)
                count+=1
            for p in a['prefixes']:assert_reproduces(inspect(a['model'],a['sequence'],p['cut'],5),p);prefixes+=1
        self.assertEqual((count,prefixes),(114,40))
    def test_independent_70digit_exhaustive_state_paths(self):
        with localcontext() as context:
            context.prec=70
            m={'version':'chainlab_v1','states':['A','B'],'symbols':['x','y'],'initial':[.6,.4],'transition':[[.7,.3],[.2,.8]],'emission':[[.8,.2],[.1,.9]]}
            for length in range(1,6):
                for obs in itertools.product(range(2),repeat=length):
                    seq=[m['symbols'][s] for s in obs];r=inspect(m,seq,length,3);paths=[]
                    for states in itertools.product(range(2),repeat=length):
                        p=D.from_float(m['initial'][states[0]])
                        for t,(state,symbol) in enumerate(zip(states,obs)):
                            if t:p*=D.from_float(m['transition'][states[t-1]][state])
                            p*=D.from_float(m['emission'][state][symbol])
                        paths.append((states,p))
                    total=sum(p for _,p in paths)
                    self.assertAlmostEqual(r['smoothing']['log_likelihood'],math.log(float(total)),places=12)
                    for t in range(length):
                        for i in range(2):self.assertAlmostEqual(r['smoothing']['state_posteriors'][t][i],float(sum(p for path,p in paths if path[t]==i)/total),places=12)
                    self.assertAlmostEqual(r['viterbi']['log_joint_probability'],math.log(float(max(p for _,p in paths))),places=12)
                    if length>1:
                        for t in range(length-1):
                            for i in range(2):
                                for j in range(2):self.assertAlmostEqual(r['smoothing']['transition_posteriors'][t][i][j],float(sum(p for path,p in paths if path[t:t+2]==(i,j))/total),places=12)
    def test_suffix_cannot_change_prefix_forecast(self):
        a=saved()['arms']['trained'];seq=a['sequence'];before=inspect(a['model'],seq,6,8)
        for i in range(6,len(seq)):
            edited=list(seq);edited[i]=a['model']['symbols'][(a['model']['symbols'].index(seq[i])+1)%3];after=inspect(a['model'],edited,6,8)
            self.assertEqual(before['prefix_forecast'],after['prefix_forecast']);self.assertEqual(before['filtering']['filtered_state_posteriors'][:6],after['filtering']['filtered_state_posteriors'][:6])
    def test_exact_counterexamples_stay_visible(self):
        r=saved()['arms'];f=r['future']['baseline'];self.assertAlmostEqual(f['filtering']['filtered_state_posteriors'][0][0],2/3);self.assertEqual(f['smoothing']['state_posteriors'][0][0],0)
        p=r['path']['baseline'];self.assertEqual(p['smoothing']['posterior_modes'],['A','A']);self.assertEqual(p['viterbi']['path'],['B','A'])
        m=r['future']['model'];x=inspect(m,['x'],1,2);one,two=[s['symbol_probabilities'][1] for s in x['prefix_forecast']['forecasts']];self.assertAlmostEqual(one,1/6);self.assertAlmostEqual(two,1/6)
        joint=math.exp(inspect(m,['x','y','y'],1)['filtering']['log_likelihood']-x['filtering']['log_likelihood']);self.assertAlmostEqual(joint,1/12);self.assertNotAlmostEqual(one*two,joint)
    def test_strict_schema_cut_horizon_and_nonmutation(self):
        a=saved()['arms']['trained'];m=a['model'];seq=a['sequence'];old=copy.deepcopy((m,seq));inspect(m,seq,1,32);self.assertEqual((m,seq),old)
        for cut in [0,13,True,1.5]:
            with self.assertRaises(ValueError):inspect(m,seq,cut)
        for horizon in [0,33,True,1.5]:
            with self.assertRaises(ValueError):inspect(m,seq,1,horizon)
        for bad in [[],['unknown'],[False]]:
            with self.assertRaises(ValueError):inspect(m,bad,1)
    def test_cli_duplicate_keys_and_valid_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'request.json';p.write_text('{"model":{},"sequence":[],"sequence":[]}');r=subprocess.run([sys.executable,'-m','chainlab','evidence','--input',str(p),'--cut','1'],capture_output=True,text=True);self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'');self.assertIn('duplicate',r.stderr)
            a=saved()['arms']['trained'];p.write_text(json.dumps(dict(model=a['model'],sequence=a['sequence'])));r=subprocess.run([sys.executable,'-m','chainlab','evidence','--input',str(p),'--cut','6'],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['cut'],6)
