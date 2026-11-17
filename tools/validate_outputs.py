"""Save real ONNX/RKNN raw outputs, or compare recordings from two devices.

No emulated inference path. Use ONNX on PC and RKNNLite on the actual board.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from cones.detector import Detector,preprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video');p.add_argument('--model');p.add_argument('--backend',choices=['onnx','rknn'])
    p.add_argument('--output');p.add_argument('--compare',nargs=2,metavar=('ONNX_NPZ','RKNN_NPZ'))
    a=p.parse_args()
    if a.compare:
        left,right=map(np.load,a.compare);records={}
        if set(left.files)!=set(right.files):raise ValueError('Recording keys differ')
        for key in left.files:
            x,y=left[key],right[key]
            if x.shape!=y.shape:raise ValueError(f'Shape mismatch: {key}')
            records[key]={'max_abs':float(np.max(np.abs(x-y))),
                          'mean_abs':float(np.mean(np.abs(x-y))),
                          'cosine':float(np.dot(x.ravel(),y.ravel())/max(np.linalg.norm(x)*np.linalg.norm(y),1e-12))}
        print(json.dumps(records,indent=2));return
    if not all((a.video,a.model,a.backend,a.output)):p.error('Recording requires --video --model --backend --output')
    cap=cv2.VideoCapture(a.video);d=Detector(a.model,a.backend);arrays={}
    try:
        for index in [0,120,360,600]:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
            if not ok:raise ValueError(f'Cannot read frame {index}')
            rgb,_=preprocess(frame)
            if a.backend=='rknn':out=d.engine.inference(inputs=[rgb[None]],data_format=['nhwc'])
            else:out=d.engine.run(None,{'images':rgb.transpose(2,0,1)[None].astype(np.float32)})
            for i,value in enumerate(out):arrays[f'frame{index}_head{i}']=value
        Path(a.output).parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**arrays)
        print('Recorded real',a.backend,'outputs:',a.output)
    finally:cap.release();d.close()


if __name__=='__main__':main()
