import argparse
import json
import sys
from .core import fit,infer,viterbi
from .demo import run_demo


def main(argv=None):
    parser=argparse.ArgumentParser(description='ChainLab discrete HMM inference, path decoding and EM')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('infer','viterbi','fit'):
        cmd=sub.add_parser(name);cmd.add_argument('--input',required=True)
        if name=='fit':cmd.add_argument('--iterations',type=int,default=25);cmd.add_argument('--tolerance',type=float,default=1e-6)
    sub.add_parser('demo');args=parser.parse_args(argv)
    try:
        if args.command=='demo':result=run_demo()
        else:
            if args.input=='-':payload=json.load(sys.stdin)
            else:
                with open(args.input,encoding='utf-8') as source:payload=json.load(source)
            field='sequences' if args.command=='fit' else 'sequence'
            if not isinstance(payload,dict) or set(payload)!={'model',field}:raise ValueError('request requires model and '+field+' only')
            if args.command=='fit':result=fit(payload['model'],payload[field],args.iterations,args.tolerance)
            elif args.command=='infer':result=infer(payload['model'],payload[field])
            else:result=viterbi(payload['model'],payload[field])
        text=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)
    except (ValueError,TypeError,OverflowError,OSError) as error:
        print('error: '+str(error),file=sys.stderr);return 2
    print(text);return 0
