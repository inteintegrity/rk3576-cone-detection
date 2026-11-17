"""Single-operator LAN FastAPI demo. Run one uvicorn worker."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from threading import Lock
from uuid import uuid4
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from cones.pipeline import process
from cones.profiles import PROFILES

ROOT=Path(__file__).resolve().parent
DATA=Path(os.environ.get('CONE_DATA',ROOT/'data')).resolve();DATA.mkdir(parents=True,exist_ok=True)
MODEL=os.environ.get('CONE_MODEL',str(ROOT/'models/cone_yolox_rk3576_fp16.rknn'))
BACKEND=os.environ.get('CONE_BACKEND','rknn')
app=FastAPI(title='RK3576 Cone Detection')
pool=ThreadPoolExecutor(max_workers=1)
lock=Lock();jobs={};active=False


@app.get('/')
def home():return FileResponse(ROOT/'web/index.html')


@app.get('/api/health')
def health():
    with lock:
        verified=any(j.get('backend')=='rknn' and j.get('frames',0)>0 for j in jobs.values())
        busy=active
    return {'backend':BACKEND,'model_exists':Path(MODEL).is_file(),'busy':busy,
            'npu_verified':verified,'note':'NPU verification requires a successful RKNN inference in this server process.'}


def run_job(job_id,source,directory,profile):
    global active
    def update(values):
        with lock:jobs[job_id].update(values)
    try:process(source,directory,MODEL,BACKEND,update=update,profile=profile)
    except Exception as exc:update({'status':'failed','error':str(exc)})
    finally:
        source.unlink(missing_ok=True)
        with lock:active=False


@app.post('/api/jobs',status_code=202)
async def upload(video:UploadFile=File(...),profile:str=Form('official-onboard')):
    global active
    if not Path(MODEL).is_file():raise HTTPException(503,'Model file is missing')
    if profile not in PROFILES:raise HTTPException(400,'Unknown camera profile')
    suffix=Path(video.filename or '').suffix.lower()
    if suffix not in {'.mp4','.mov','.avi','.webm','.mkv'}:raise HTTPException(400,'Choose a video file')
    with lock:
        if active:raise HTTPException(409,'Another video is being processed')
        active=True
    job_id=uuid4().hex;directory=DATA/job_id;directory.mkdir();source=directory/('upload'+suffix)
    try:
        size=0
        with source.open('wb') as out:
            while chunk:=await video.read(1024*1024):
                size+=len(chunk)
                if size>512*1024*1024:raise HTTPException(413,'Upload limit is 512 MB')
                out.write(chunk)
        if not size:raise HTTPException(400,'Empty video')
        with lock:jobs[job_id]={'id':job_id,'status':'queued','backend':BACKEND,'frames':0,'counts':{},'profile':profile}
        pool.submit(run_job,job_id,source,directory,profile)
        return {'id':job_id}
    except BaseException:
        source.unlink(missing_ok=True)
        with lock:active=False
        raise
    finally:await video.close()


@app.get('/api/jobs/{job_id}')
def status(job_id:str):
    with lock:
        if job_id not in jobs:raise HTTPException(404,'Job not found')
        return dict(jobs[job_id])


@app.get('/api/jobs/{job_id}/{kind}')
def artifact(job_id:str,kind:str):
    with lock:
        if job_id not in jobs:raise HTTPException(404,'Job not found')
        state=jobs[job_id]['status']
    names={'preview':'preview.jpg','video':'result.mp4','summary':'summary.json','detections':'detections.jsonl'}
    if kind not in names:raise HTTPException(404,'Artifact not found')
    if kind!='preview' and state!='completed':raise HTTPException(409,'Result is not ready')
    path=DATA/job_id/names[kind]
    if not path.is_file():raise HTTPException(404,'Artifact is not ready')
    return FileResponse(path,filename=names[kind] if kind!='preview' else None,
                        headers={'Cache-Control':'no-store'})
