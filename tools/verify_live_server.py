"""Upload an entire real video to a running server and verify its artifacts.

Requires httpx on the client. Uses the selected real server backend, no mocks.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import httpx

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
p.add_argument('--profile',choices=['official-onboard','none'],default='official-onboard')
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
with httpx.Client(base_url=a.url.rstrip('/'),timeout=120,trust_env=False) as c:
    with open(a.input,'rb') as f:
        r=c.post('/api/jobs',files={'video':(Path(a.input).name,f,'video/mp4')},data={'profile':a.profile})
    r.raise_for_status();job=r.json()['id'];deadline=time.monotonic()+600
    (out/'job.json').write_text(json.dumps({'id':job,'url':a.url+'/#'+job},indent=2))
    while time.monotonic()<deadline:
        r=c.get('/api/jobs/'+job);r.raise_for_status();j=r.json()
        if j['status'] in ('completed','failed'):break
        time.sleep(1)
    if j['status']!='completed':raise RuntimeError(j)
    for kind,name in [('summary','summary.json'),('detections','detections.jsonl'),('preview','preview.jpg'),('video','result.mp4')]:
        r=c.get(f'/api/jobs/{job}/{kind}');r.raise_for_status();(out/name).write_bytes(r.content)
    r=c.get(f'/api/jobs/{job}/video',headers={'Range':'bytes=0-1023'})
    assert r.status_code==206 and len(r.content)==1024,'MP4 range serving failed'
    summary=json.loads((out/'summary.json').read_text())
    trace=[json.loads(s) for s in (out/'detections.jsonl').read_text().splitlines()]
    assert len(trace)==summary['frames']
    cap=cv2.VideoCapture(str(out/'result.mp4'));assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==summary['frames'];cap.release()
    assert summary['profile']==a.profile
    if summary['backend']=='rknn':assert summary['npu_fps']>0
    assert sum(len(f['excluded_detections']) for f in trace)==summary['excluded_observations']
    (out/'health.json').write_text(json.dumps(c.get('/api/health').json(),indent=2))
    print(json.dumps(summary,indent=2));print('Verified result:',a.url+'/#'+job)
