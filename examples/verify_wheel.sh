#!/usr/bin/env bash
set -euo pipefail
installed=${1}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export PYTHONPATH="$installed"
python - <<'PY'
import importlib,pkgutil,chainlab,json
from chainlab.evidence_export import saved,build,assert_reproduces
mods=list(pkgutil.walk_packages(chainlab.__path__,chainlab.__name__+'.'))
for m in mods:importlib.import_module(m.name)
assert len(mods)==10
r=saved();assert_reproduces(build(),r)
a=r['arms']['trained'];json.dump(dict(model=a['model'],sequence=a['sequence']),open('request.json','w'));json.dump(dict(model=a['model'],sequences=[a['sequence']]),open('fit.json','w'))
print('all11modules,114edits40prefixes and original training/online reports reproduce')
PY
for command in infer viterbi filter;do python -m chainlab "$command" --input request.json > "$command.json";done
python -m chainlab forecast --input request.json --horizon 3 > forecast.json
python -m chainlab fit --input fit.json --iterations 2 > fit-out.json
python -m chainlab evidence --input request.json --cut 6 > evidence.json
python -m chainlab evidence-report > evidence-report.json
python -m chainlab demo > demo.json
python -m chainlab online-demo > online-demo.json
python - <<'PY'
import json
from chainlab.evidence_export import saved,assert_reproduces
r=saved();assert_reproduces(json.load(open('demo.json')),r['original_demo']);assert_reproduces(json.load(open('online-demo.json')),r['original_online_demo']);assert_reproduces(json.load(open('evidence-report.json')),r);assert_reproduces(json.load(open('evidence.json')),r['arms']['trained']['baseline'])
assert len(json.load(open('infer.json'))['state_posteriors'])==12
assert len(json.load(open('viterbi.json'))['path'])==12
assert len(json.load(open('filter.json'))['filtered_state_posteriors'])==12
assert len(json.load(open('forecast.json'))['forecasts'])==3
assert json.load(open('fit-out.json'))['diagnostics']['observations']==12
print('all9commands reproduce outside checkout')
PY

python - <<'PY2'
import chainlab,json,subprocess,sys,copy
from pathlib import Path
from chainlab.pathrisk import audit
from chainlab.path_verify import check
root=Path(chainlab.__file__).parent
controls=[c for kind in ['core','rounded'] for c in json.loads((root/('artifacts/path-controls-'+kind+'.json')).read_text())]
for c in controls:
    r=audit(c['input']);assert r==c['report'];assert check(r)
    Path('path-input.json').write_text(json.dumps(c['input']))
    out=subprocess.run([sys.executable,'-m','chainlab.pathrisk','path-input.json'],capture_output=True,text=True);assert out.returncode==0,out.stderr
    Path('path-report.json').write_text(out.stdout)
    v=subprocess.run([sys.executable,'-m','chainlab.path_verify','path-report.json'],capture_output=True,text=True);assert v.returncode==0,v.stderr
    for key in ['evidence','supported_paths']:
        bad=copy.deepcopy(r);bad[key]='invalid';assert subprocess.run([sys.executable,'-m','chainlab.path_verify','-'],input=json.dumps(bad),capture_output=True,text=True).returncode==2
print('13 exact path controls, both module commands and26 forgeries checked outside checkout')
PY2

python - <<'PY3'
import json
from fractions import Fraction as F
from pathlib import Path
import chainlab
from chainlab.pathrisk import audit
from chainlab.occupancy import visit_distribution
root=Path(chainlab.__file__).parent
controls=[c for kind in ['core','rounded'] for c in json.loads((root/('artifacts/path-controls-'+kind+'.json')).read_text())]
for c in controls:
    inp=c['input'];name=inp['model']['states'][0];r=visit_distribution(inp,[name])
    assert F(r['evidence'])==F(c['report']['evidence'])
    ps=list(map(F,r['posterior_count_probabilities']));assert sum(ps)==1
    assert F(r['expected_count'])==sum(F(row[0]) for row in c['report']['state'])
    other=visit_distribution(inp,inp['model']['states'][1:])
    assert r['posterior_count_probabilities']==other['posterior_count_probabilities'][::-1]
by={c['name']:c['input'] for c in controls}
a=visit_distribution(by['independent'],[by['independent']['model']['states'][0]])
b=visit_distribution(by['locked'],[by['locked']['model']['states'][0]])
assert a['expected_count']==b['expected_count']=='5/2'
assert a['count_variance']=='5/4' and b['count_variance']=='25/4'
assert a['posterior_count_probabilities']==['1/32','5/32','5/16','5/16','5/32','1/32']
assert b['posterior_count_probabilities']==['1/2','0','0','0','0','1/2']
print('13 visit-count laws and complements; equal marginals/different count variance checked outside checkout')
PY3
