"""Compile the extracted official YOLOX ONNX for RK3576 (Linux Toolkit2)."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from rknn.api import RKNN


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx',required=True);p.add_argument('--output',required=True)
    p.add_argument('--dataset',help='Optional representative calibration image list; enables INT8')
    a=p.parse_args();Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    r=RKNN(verbose=False)
    try:
        # Raw RGB 0..255. ImageNet zscore is already in the ONNX graph.
        r.config(target_platform='rk3576',mean_values=[[0,0,0]],std_values=[[1,1,1]])
        for name,call in [('load',lambda:r.load_onnx(model=a.onnx)),
                          ('build',lambda:r.build(do_quantization=bool(a.dataset),dataset=a.dataset)),
                          ('export',lambda:r.export_rknn(a.output))]:
            result=call()
            if result!=0:raise RuntimeError(f'{name} failed: {result}')
        record={'target':'rk3576','toolkit':importlib.metadata.version('rknn-toolkit2'),
                'precision':'int8' if a.dataset else 'fp16','onnx_sha256':hashlib.sha256(Path(a.onnx).read_bytes()).hexdigest(),
                'rknn_sha256':hashlib.sha256(Path(a.output).read_bytes()).hexdigest()}
        Path(a.output).with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2))
    finally:r.release()


if __name__=='__main__':main()
