import argparse
import json
import sys
from .core import fit,infer,viterbi
from .demo import run_demo,run_online_demo
from .online import filter_sequence,forecast
from .evidence import inspect
from .evidence_export import saved


def _unique(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('duplicate JSON key: '+k)
        result[k]=v
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description='ChainLab discrete HMM inference, path decoding and EM')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('infer','viterbi','fit','filter','forecast','evidence'):
        cmd=sub.add_parser(name);cmd.add_argument('--input',required=True)
        if name=='evidence':cmd.add_argument('--cut',type=int,required=True);cmd.add_argument('--horizon',type=int,default=5)
        if name=='forecast':cmd.add_argument('--horizon',type=int,default=1)
        if name=='fit':cmd.add_argument('--iterations',type=int,default=25);cmd.add_argument('--tolerance',type=float,default=1e-6)
    sub.add_parser('evidence-report');sub.add_parser('demo');sub.add_parser('online-demo');args=parser.parse_args(argv)
    try:
        if args.command=='evidence-report':result=saved()
        elif args.command=='demo':result=run_demo()
        elif args.command=='online-demo':result=run_online_demo()
        else:
            if args.input=='-':payload=json.load(sys.stdin,object_pairs_hook=_unique)
            else:
                with open(args.input,encoding='utf-8') as source:payload=json.load(source,object_pairs_hook=_unique)
            field='sequences' if args.command=='fit' else 'sequence'
            if not isinstance(payload,dict) or set(payload)!={'model',field}:raise ValueError('request requires model and '+field+' only')
            if args.command=='evidence':result=inspect(payload['model'],payload[field],args.cut,args.horizon)
            elif args.command=='fit':result=fit(payload['model'],payload[field],args.iterations,args.tolerance)
            elif args.command=='infer':result=infer(payload['model'],payload[field])
            elif args.command=='filter':result=filter_sequence(payload['model'],payload[field])
            elif args.command=='forecast':result=forecast(payload['model'],payload[field],args.horizon)
            else:result=viterbi(payload['model'],payload[field])
        text=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)
    except (ValueError,TypeError,OverflowError,OSError) as error:
        print('error: '+str(error),file=sys.stderr);return 2
    print(text);return 0
