"""Reproduce original fixtures and independently checkable observation-edit witnesses."""
import json,math
from pathlib import Path
from .demo import run_demo,run_online_demo,fixture
from .evidence import inspect

def build():
    original=run_demo();online=run_online_demo();truth,initial,train,test=fixture()
    arms={}
    models={'trained':original['training']['model'],'initial':initial,'reference':truth,
            'future':online['future_information_counterexample']['model'],
            'path':original['posterior_mode_counterexample']['model']}
    for name,model in models.items():
        sequence=test[0][:12] if name in ['trained','initial','reference'] else (['x','y'] if name=='future' else ['x','x'])
        cut=min(6,len(sequence));base=inspect(model,sequence,cut,5);edits=[]
        for position in range(len(sequence)):
            for symbol in model['symbols']:
                changed=list(sequence);changed[position]=symbol
                try:result=inspect(model,changed,cut,5)
                except ValueError as error:result={'error':str(error)}
                edits.append(dict(position=position,symbol=symbol,result=result))
        arms[name]=dict(model=model,sequence=sequence,baseline=base,
                        prefixes=[inspect(model,sequence,c,5) for c in range(1,len(sequence)+1)],edits=edits)
    return dict(version=1,original_demo=original,original_online_demo=online,original_first_test_sequence=test[0],arms=arms)

def assert_reproduces(a,b,path='root'):
    if isinstance(b,dict):
        assert isinstance(a,dict) and a.keys()==b.keys(),path
        for k in b:assert_reproduces(a[k],b[k],path+'.'+k)
    elif isinstance(b,list):
        assert isinstance(a,list) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):assert_reproduces(x,y,path+'['+str(i)+']')
    elif type(b) is float:assert type(a) in (int,float) and math.isfinite(a) and math.isclose(a,b,abs_tol=1e-11,rel_tol=1e-12),(path,a,b)
    else:assert type(a) is type(b) and a==b,(path,a,b)

def encoded(x):return json.dumps(x,separators=(',',':'),allow_nan=False)+'\n'
def export(root):
    root=Path(root);(root/'chainlab/artifacts').mkdir(exist_ok=True);(root/'docs').mkdir(exist_ok=True);r=build()
    (root/'chainlab/artifacts/evidence-meta.json').write_text(encoded({k:v for k,v in r.items() if k!='arms'}))
    for name,a in r['arms'].items():
        (root/('chainlab/artifacts/evidence-'+name+'.json')).write_text(encoded({k:v for k,v in a.items() if k not in ['edits','prefixes']}))
        for key in ['edits','prefixes']:
            for j in range(0,len(a[key]),12):(root/('chainlab/artifacts/evidence-'+name+'-'+key+'-'+str(j//12)+'.json')).write_text(encoded(a[key][j:j+12]))
    public=dict(version=1,original_demo=r['original_demo'],original_online_demo=r['original_online_demo'],original_first_test_sequence=r['original_first_test_sequence'],arms={k:{x:a[x] for x in ['model','sequence']} for k,a in r['arms'].items()})
    (root/'docs/data.js').write_text('globalThis.CHAIN_DATA='+encoded(public).rstrip()+';\n')
    return dict(arms=len(r['arms']),edit_witnesses=sum(len(a['edits']) for a in r['arms'].values()),prefix_witnesses=sum(len(a['prefixes']) for a in r['arms'].values()))

def saved():
    root=Path(__file__).parent/'artifacts';r=json.loads((root/'evidence-meta.json').read_text());r['arms']={}
    for name in ['trained','initial','reference','future','path']:
        a=json.loads((root/('evidence-'+name+'.json')).read_text())
        for key in ['edits','prefixes']:a[key]=[v for p in sorted(root.glob('evidence-'+name+'-'+key+'-*.json')) for v in json.loads(p.read_text())]
        r['arms'][name]=a
    return r
if __name__=='__main__':print(json.dumps(export(Path(__file__).resolve().parents[1])))
