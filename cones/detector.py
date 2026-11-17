"""Official four-class YOLOX preprocessing, decoding and actual backends."""
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np

CLASSES = ('greenCone', 'redCone', 'yellowConeLeft', 'yellowConeRight')
COLORS = ((80,220,60), (70,70,245), (40,220,250), (10,160,255))  # BGR


def preprocess(frame):
    h,w = frame.shape[:2]
    scale = min(416/w,416/h)
    nw,nh = max(1,round(w*scale)),max(1,round(h*scale))
    x,y = (416-nw)//2,(416-nh)//2
    image = np.full((416,416,3),114,np.uint8)
    image[y:y+nh,x:x+nw] = cv2.resize(frame,(nw,nh))[:,:,::-1]
    return image,(nw/w,nh/h,x,y,w,h)


def decode(outputs, transform, confidence=.35, iou=.45):
    predictions=[]
    for out,stride in zip(outputs,(8,16,32)):
        a=np.asarray(out)
        if a.shape != (1,9,416//stride,416//stride):
            raise ValueError(f'Unexpected YOLOX output {a.shape}')
        a=a[0].transpose(1,2,0).reshape(-1,9)
        probabilities=1/(1+np.exp(-np.clip(a[:,4:],-80,80)))
        cls=probabilities[:,1:].argmax(1)
        scores=probabilities[:,0]*probabilities[np.arange(len(a)),cls+1]
        keep=scores>=confidence
        gy,gx=np.mgrid[:416//stride,:416//stride]
        centers=(a[:,:2]+np.stack((gx,gy),-1).reshape(-1,2))*stride
        sizes=np.exp(np.clip(a[:,2:4],-20,20))*stride
        boxes=np.concatenate((centers-sizes/2,sizes),1)
        predictions.extend(zip(boxes[keep],scores[keep],cls[keep]))
    sx,sy,px,py,w,h=transform
    detections=[]
    for category in range(len(CLASSES)):
        candidates=[p for p in predictions if p[2]==category]
        if not candidates: continue
        indices=cv2.dnn.NMSBoxes([p[0].tolist() for p in candidates],
                                 [float(p[1]) for p in candidates],confidence,iou)
        for i in np.asarray(indices).reshape(-1):
            box,score,_=candidates[int(i)]
            x1,y1,bw,bh=map(float,box)
            box=[float(np.clip((x1-px)/sx,0,w)),float(np.clip((y1-py)/sy,0,h)),
                 float(np.clip((x1+bw-px)/sx,0,w)),float(np.clip((y1+bh-py)/sy,0,h))]
            if box[2]>box[0] and box[3]>box[1]:
                detections.append({'class_id':category,'label':CLASSES[category],
                                   'confidence':float(score),'box':box})
    return sorted(detections,key=lambda d:d['confidence'],reverse=True)


class Detector:
    def __init__(self, model, backend='rknn', confidence=.35):
        self.backend,self.confidence=backend,confidence
        if not Path(model).is_file(): raise FileNotFoundError(model)
        if backend=='rknn':
            from rknnlite.api import RKNNLite
            self.engine=RKNNLite()
            try:
                if self.engine.load_rknn(str(model))!=0: raise RuntimeError('RKNN model load failed')
                if self.engine.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)!=0:
                    raise RuntimeError('RK3576 NPU initialization failed')
            except Exception:
                self.engine.release();raise
        elif backend=='onnx':
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=4
            self.engine=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
        else: raise ValueError('backend must be rknn or onnx')

    def infer(self,frame):
        rgb,transform=preprocess(frame)
        start=perf_counter()
        if self.backend=='rknn':
            outputs=self.engine.inference(inputs=[rgb[None]],data_format=['nhwc'])
        else:
            outputs=self.engine.run(None,{'images':rgb.transpose(2,0,1)[None].astype(np.float32)})
        seconds=perf_counter()-start
        if outputs is None: raise RuntimeError('Inference returned no outputs')
        return decode(outputs,transform,self.confidence),seconds

    def close(self):
        if self.backend=='rknn':self.engine.release()


def annotate(frame,detections,counts,npu_fps,e2e_fps,backend):
    frame=frame.copy()
    for d in detections:
        x1,y1,x2,y2=map(round,d['box']);color=COLORS[d['class_id']]
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        text=f"{d['label']} {d['confidence']:.2f}"
        cv2.putText(frame,text,(x1,max(12,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
    overlay=frame.copy();cv2.rectangle(overlay,(0,0),(frame.shape[1],58),(18,24,32),-1)
    frame=cv2.addWeighted(overlay,.88,frame,.12,0)
    speed=f'NPU API: {npu_fps:.1f} FPS' if backend=='rknn' else 'CPU ONNX (no NPU)'
    cv2.putText(frame,f'{speed} | Processing: {e2e_fps:.1f} FPS',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.42,(245,245,245),1,cv2.LINE_AA)
    text=f"Green {counts[CLASSES[0]]}   Red {counts[CLASSES[1]]}   Yellow L {counts[CLASSES[2]]} / R {counts[CLASSES[3]]}"
    cv2.putText(frame,text,(8,44),cv2.FONT_HERSHEY_SIMPLEX,.42,(245,245,245),1,cv2.LINE_AA)
    return frame
