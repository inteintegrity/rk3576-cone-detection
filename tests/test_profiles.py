import pytest
from cones.profiles import exclude_vehicle_markers


def detection(box,label='redCone'):
    return {'box':box,'class_id':1,'label':label,'confidence':.9}


def test_both_vehicle_flags_removed_regardless_of_predicted_color():
    flags=[detection([103.8,372,138.2,415.5]),detection([268.2,374.8,315.2,413.2],'yellowConeLeft'),
           detection([192.6,357.5,306,415.5],'yellowConeLeft')]
    kept,excluded=exclude_vehicle_markers(flags,(416,416,3),'official-onboard')
    assert kept==[] and len(excluded)==3
    assert flags[0].get('exclusion_reason') is None


def test_nearby_road_cones_kept():
    # Real adjacent cone from frame 360 plus fixtures just above/alongside masks.
    road=[detection([381.4,278.1,410.9,339.3]),detection([300,225,335,275]),
          detection([50,365,70,415]),detection([180,365,200,415])]
    kept,excluded=exclude_vehicle_markers(road,(416,416,3),'official-onboard')
    assert kept==road and excluded==[]


def test_none_keeps_original_predictions():
    raw=[detection([268,374,315,413])]
    assert exclude_vehicle_markers(raw,(416,416,3),'none')==(raw,[])


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):exclude_vehicle_markers([],(416,416,3),'unknown')
