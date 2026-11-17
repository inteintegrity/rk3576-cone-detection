"""Video jobs with bounded memory and measured inference/encoding times."""
import json
import shutil
import subprocess
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from .detector import CLASSES, Detector, annotate
from .profiles import PROFILES,exclude_vehicle_markers


def process(source, directory, model, backend='rknn', confidence=.35, update=None, profile='official-onboard'):
    if profile not in PROFILES:raise ValueError('Unknown camera profile: '+profile)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(source));detector=None;writer=None
    try:
        if not cap.isOpened():raise ValueError('Cannot decode this video')
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<=0:fps=30.
        w,h=int(cap.get(3)),int(cap.get(4));total=int(cap.get(7))
        if w<2 or h<2 or w*h>3840*2160:raise ValueError('Video must be between 2x2 and 3840x2160')
        # yuv420p output requires even dimensions.
        w,h=w-w%2,h-h%2
        detector=Detector(model,backend,confidence)
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        ok,first=cap.read()
        if not ok:raise ValueError('Video has no decodable frames')
        # Warmup is real inference, excluded from steady-state NPU statistics.
        for _ in range(5):detector.infer(first[:h,:w])
        if not shutil.which('ffmpeg'):raise RuntimeError('ffmpeg is required for browser-compatible H.264 MP4')
        intermediate=directory/'intermediate.mp4'
        writer=cv2.VideoWriter(str(intermediate),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        if not writer.isOpened():raise RuntimeError('Cannot open MP4 writer')
        start=perf_counter();inference_times=[];count=0;inference_total=0.;excluded_total=0
        sums=dict.fromkeys(CLASSES,0);peaks=sums.copy();counts=sums.copy()
        with (directory/'detections.jsonl').open('w') as trace:
            frame=first
            while frame is not None:
                frame=frame[:h,:w]
                raw,seconds=detector.infer(frame);inference_times.append(seconds);inference_total+=seconds
                detections,excluded=exclude_vehicle_markers(raw,frame.shape,profile);excluded_total+=len(excluded)
                counts={name:sum(d['label']==name for d in detections) for name in CLASSES}
                for name in CLASSES:sums[name]+=counts[name];peaks[name]=max(peaks[name],counts[name])
                count+=1
                npu_fps=count/inference_total if backend=='rknn' else None
                current_fps=count/max(perf_counter()-start,1e-9)
                drawn=annotate(frame,detections,counts,npu_fps or 0,current_fps,backend)
                writer.write(drawn)
                trace.write(json.dumps({'frame':count-1,'time_s':(count-1)/fps,'inference_ms':seconds*1000,
                                        'counts':counts,'detections':detections,'raw_detections':raw,
                                        'excluded_detections':excluded})+'\n')
                if count==1 or count%10==0:
                    ok,jpeg=cv2.imencode('.jpg',drawn,[cv2.IMWRITE_JPEG_QUALITY,85])
                    if ok:
                        temp=directory/'preview.tmp';temp.write_bytes(jpeg.tobytes());temp.replace(directory/'preview.jpg')
                    if update:update({'status':'processing','frames':count,'total_frames':total,'counts':counts,
                                      'npu_fps':npu_fps,'processing_fps':current_fps,'profile':profile,
                                      'excluded_observations':excluded_total})
                ok,frame=cap.read()
                if not ok:frame=None
        writer.release();writer=None
        pipeline_seconds=perf_counter()-start
        if count==0:raise ValueError('No frames processed')
        if update:update({'status':'encoding','frames':count})
        encode_start=perf_counter()
        result=subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-i',str(intermediate),
            '-an','-c:v','libx264','-preset','veryfast','-crf','21','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(directory/'result.mp4')],capture_output=True,text=True)
        if result.returncode:raise RuntimeError('H.264 encoding failed: '+result.stderr[-2000:])
        encoding_seconds=perf_counter()-encode_start
        elapsed=perf_counter()-start
        intermediate.unlink()
        summary={'status':'completed','backend':backend,'frames':count,'total_frames':total,
                 'source_fps':fps,'video_seconds':count/fps,'counts':counts,
                 'peak_counts':peaks,'detection_observations':sums,
                 'count_definition':'Per-frame detections; observations sum repeated appearances, not unique cones.',
                 'npu_fps':count/sum(inference_times) if backend=='rknn' else None,
                 'inference_mean_ms':float(np.mean(inference_times)*1000),
                 'inference_p50_ms':float(np.median(inference_times)*1000),
                 'inference_p95_ms':float(np.percentile(inference_times,95)*1000),
                 'processing_fps':count/pipeline_seconds,'end_to_end_fps':count/elapsed,
                 'pipeline_seconds':pipeline_seconds,'encoding_seconds':encoding_seconds,'elapsed_seconds':elapsed,
                 'warmup_frames':5,'confidence':confidence,'profile':profile,
                 'exclusion_polygons_normalized':PROFILES[profile],'excluded_observations':excluded_total,
                 'timing_definition':'NPU FPS = frames / synchronous RKNNLite.inference wall time (includes transfer/runtime overhead). E2E includes decode, preprocess, inference, NMS, draw, trace, video write and final H.264 encode; excludes upload/model load/warmup. No device-cycle claim.'}
        (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if update:update(summary)
        return summary
    finally:
        cap.release()
        if writer is not None:writer.release()
        if detector is not None:detector.close()
