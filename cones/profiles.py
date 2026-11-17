"""Camera-specific ego-marker masks. Raw network predictions are retained.

The official fixed onboard camera sees two red triangular markings on the car's
lower side panels. They are not track cones. A different camera must use `none`
or supply its own normalized polygons after checking the physical camera view.
"""
import cv2
import numpy as np

PROFILES={
    'none':[],
    'official-onboard':[
        [[.195,.885],[.325,.885],[.345,1.0],[.185,1.0]],
        [[.56,.885],[.765,.885],[.805,1.0],[.56,1.0]],
    ],
}


def exclude_vehicle_markers(detections,shape,profile):
    if profile not in PROFILES:raise ValueError('Unknown camera profile: '+profile)
    h,w=shape[:2]
    polygons=[np.asarray(points,np.float32)*[w,h] for points in PROFILES[profile]]
    kept,excluded=[],[]
    for detection in detections:
        x1,y1,x2,y2=detection['box'];center=((x1+x2)/2,(y1+y2)/2)
        if any(cv2.pointPolygonTest(p.astype(np.float32),center,False)>=0 for p in polygons):
            excluded.append({**detection,'exclusion_reason':'ego_vehicle_marker_region'})
        else:kept.append(detection)
    return kept,excluded
