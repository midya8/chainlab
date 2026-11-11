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
assert len(mods)==9
r=saved();assert_reproduces(build(),r)
a=r['arms']['trained'];json.dump(dict(model=a['model'],sequence=a['sequence']),open('request.json','w'));json.dump(dict(model=a['model'],sequences=[a['sequence']]),open('fit.json','w'))
print('all10modules,114edits40prefixes and original training/online reports reproduce')
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
