"""Deterministic decoder fixtures; these are not an inference backend."""
import numpy as np
import pytest
from cones.detector import decode,preprocess


def maps():return [np.full((1,9,s,s),-30,np.float32) for s in (52,26,13)]


def test_empty():assert decode(maps(),(1,1,0,0,416,416))==[]


def test_class_order_box_and_nms():
    outputs=maps()
    # Two overlapping green fixtures plus a red fixture at the same location:
    # class-aware NMS must retain green and red independently.
    for y,x,c in [(10,10,0),(10,11,0),(11,10,1)]:
        outputs[0][0,:4,y,x]=[10.5-x,10.5-y,np.log(2),np.log(3)]
        outputs[0][0,4,y,x]=8
        outputs[0][0,5+c,y,x]=8
    ds=decode(outputs,(1,1,0,0,416,416))
    assert {d['label'] for d in ds}=={'greenCone','redCone'}
    assert len(ds)==2
    np.testing.assert_allclose(ds[0]['box'],[76,72,92,96],atol=1e-4)


def test_letterbox_color_and_roundtrip():
    frame=np.zeros((180,320,3),np.uint8);frame[:]=[15,35,200]
    rgb,t=preprocess(frame)
    assert rgb.shape==(416,416,3)
    np.testing.assert_array_equal(rgb[208,208],[200,35,15])
    np.testing.assert_array_equal(rgb[0,0],[114,114,114])
    sx,sy,px,py,w,h=t
    assert (w,h)==(320,180)
    assert sx*320==416 and sy*180+2*py==416


def test_reject_wrong_model():
    with pytest.raises(ValueError,match='Unexpected'):
        decode([np.zeros((1,85,52,52))],(1,1,0,0,416,416))
