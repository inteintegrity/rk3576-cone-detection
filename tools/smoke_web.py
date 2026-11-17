"""Exercise upload -> real inference -> H.264 download with FastAPI TestClient."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from fastapi.testclient import TestClient

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
p.add_argument('--backend',default='onnx',choices=['onnx','rknn'])
a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
os.environ.update(CONE_MODEL=str(Path(a.model).resolve()),CONE_BACKEND=a.backend,CONE_DATA=str(out/'jobs'))
import app
clip=out/'smoke-input.mp4';cap=cv2.VideoCapture(a.input)
w,h=int(cap.get(3)),int(cap.get(4));writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'mp4v'),cap.get(5),(w,h))
for _ in range(24):
    ok,frame=cap.read()
    if not ok:raise ValueError('Input must contain at least 24 frames')
    writer.write(frame)
writer.release();cap.release()
with TestClient(app.app) as client:
    assert client.get('/').status_code==200
    assert client.post('/api/jobs',files={'video':('bad.txt',b'bad','text/plain')}).status_code==400
    assert client.post('/api/jobs',files={'video':('empty.mp4',b'','video/mp4')}).status_code==400
    with clip.open('rb') as f:r=client.post('/api/jobs',files={'video':('input.mp4',f,'video/mp4')})
    assert r.status_code==202,r.text
    job=r.json()['id'];deadline=time.monotonic()+180
    # Submission reserves the only inference worker immediately.
    with clip.open('rb') as f:
        assert client.post('/api/jobs',files={'video':('input.mp4',f,'video/mp4')}).status_code==409
    while time.monotonic()<deadline:
        status=client.get('/api/jobs/'+job).json()
        if status['status'] in ('completed','failed'):break
        time.sleep(.2)
    assert status['status']=='completed',status
    assert status['frames']==24,status
    assert (status['npu_fps'] is None)==(a.backend=='onnx')
    for kind in ['video','preview','summary','detections']:
        r=client.get(f'/api/jobs/{job}/{kind}');assert r.status_code==200,(kind,r.text)
        if kind=='video':(out/'downloaded.mp4').write_bytes(r.content)
        if kind=='detections':assert len(r.text.splitlines())==24
    assert client.get('/api/jobs/not-a-job/video').status_code==404
    cap=cv2.VideoCapture(str(out/'downloaded.mp4'));assert int(cap.get(7))==24;cap.release()
    (out/'web-smoke-summary.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))
