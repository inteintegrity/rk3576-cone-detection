import argparse
import json
from cones.pipeline import process

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Real cone detection video processing')
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',required=True);p.add_argument('--backend',choices=['rknn','onnx'],default='rknn')
    p.add_argument('--confidence',type=float,default=.35)
    p.add_argument('--profile',choices=['official-onboard','none'],default='official-onboard')
    a=p.parse_args()
    print(json.dumps(process(a.input,a.output,a.model,a.backend,a.confidence,profile=a.profile),indent=2))
