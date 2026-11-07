#!/usr/bin/env bash
set -eu
python -m chainlab demo
python -m chainlab infer --input examples/request.json
python -m chainlab viterbi --input examples/request.json
python - <<'PY'
import json
from chainlab import fit,infer,viterbi
with open('examples/request.json') as source:request=json.load(source)
report=fit(request['model'],[request['sequence'],['y','y','x']],iterations=4)
with open('/tmp/chainlab-trained-request.json','w') as target:
    json.dump({'model':report['model'],'sequence':['x','y']},target)
assert all(b['log_likelihood']>=a['log_likelihood'] for a,b in zip(report['history'],report['history'][1:]))
impossible=dict(request['model'],emission=[[1,0],[1,0]])
try:infer(impossible,['y'])
except ValueError as error:assert 'zero probability' in str(error)
else:raise AssertionError('impossible observation must reject, not smooth')
print('EM accepted likelihood is nondecreasing; impossible observations reject')
PY
python -m chainlab infer --input /tmp/chainlab-trained-request.json
PYTHONPATH=. python examples/render_demo.py
git diff --exit-code -- assets/sequence-audit.svg 2>/dev/null || { test ! -d .git; }
