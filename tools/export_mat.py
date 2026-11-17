"""Extract this project's official MATLAB v5 MCOS detector without MATLAB.

This is a deliberately narrow importer for the pinned official artifact, not a
general MATLAB object loader. Unsupported layers and changed topology fail closed.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as H, numpy_helper as NH, TensorProto as T
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader


def cells(path):
    m = loadmat(path)
    w = m['__function_workspace__'].tobytes()
    stream = m['__header__'].ljust(124, b' ') + b'\x00\x01IM' + w[8:]
    reader = MatFile5Reader(BytesIO(stream), struct_as_record=False, squeeze_me=True)
    reader.initialize_read()
    reader.read_file_header()
    header, _ = reader.read_var_header()
    return reader.read_var_array(header).MCOS[0][2]


def export(source, destination):
    expected='fcd732b620e52b8110e6c3ce869af86dc7d91a18abac2113bbee391d859275bd'
    if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=expected:
        raise ValueError('Unsupported artifact: importer requires the pinned official MAT SHA-256')
    v = cells(source)
    assert len(v) == 772, 'Unsupported MATLAB serialization'
    meta = v[768]
    classes = list(meta.ClassNames)
    assert classes == ['greenCone', 'redCone', 'yellowConeLeft', 'yellowConeRight']
    layers = {x.Name: x for x in v if hasattr(x, 'Name')}
    edges = defaultdict(list)
    for index in (341, 580, 756):
        for src, dst in zip(*v[index]):
            name, _, port = dst.partition('/')
            edges[name].append((int(port[2:]) if port.startswith('in') and port[2:].isdigit() else 0, src))
    # Public dlnetwork stages use five neck inputs, two of which are references
    # for resize. References are unnecessary with the fixed 416x416 input.
    p = 'TopLevelModule_backbone_'
    for src, dst in [
        (p+'backbone_dark5_2_conv3_act', p+'lateral_conv0_conv'),
        (p+'backbone_dark4_1_conv3_act', 'Cat21'),
        (p+'backbone_dark3_1_conv3_act', 'Cat23'),
    ]:
        edges[dst].append((2 if dst.startswith('Cat') else 0, src))
    for i, name in enumerate(v[589].OutputNames):
        edges[f'TopLevelModule_head_stems_{i}_conv'].append((0, name))
    nodes, initializers = [], []

    def constant(name, value):
        initializers.append(NH.from_array(np.asarray(value), name))
        return name

    def node(op, inputs, output, **attrs):
        nodes.append(H.make_node(op, inputs, [output], name=output, **attrs))

    # Input is RGB float32 in raw [0,255]; normalization remains in the graph.
    norm = layers['inputLayer'].Normalization
    mean = constant('rgb_mean', np.asarray(norm.Mean, np.float32).reshape(1,3,1,1))
    std = constant('rgb_std', np.asarray(norm.Std, np.float32).reshape(1,3,1,1))
    node('Sub', ['images', mean], 'centered')
    node('Div', ['centered', std], 'inputLayer')
    # YOLOX Focus: [top-left, bottom-left, top-right, bottom-right].
    for i, (y, x) in enumerate(((0,0),(1,0),(0,1),(1,1))):
        args = [constant(f'focus_{i}_{k}', np.array(a, np.int64)) for k,a in
                [('start',[y,x]),('end',[416,416]),('axis',[2,3]),('step',[2,2])]]
        node('Slice', ['inputLayer'] + args, f'focus_{i}')
    node('Concat', [f'focus_{i}' for i in range(4)], 'spaceToDepth', axis=1)
    done = {'inputLayer', 'spaceToDepth'}
    # Export three raw feature maps; decoding and NMS stay outside the NPU.
    targets = ['Cat31', 'Cat34', 'Cat37']
    needed = set()
    def visit(name):
        if name in needed or name in done:
            return
        needed.add(name)
        for _, src in edges[name]: visit(src)
    for name in targets: visit(name)
    while needed - done:
        advanced = False
        for name in sorted(needed - done):
            ins = [src for _,src in sorted(edges[name])]
            if not ins or not all(src in done for src in ins): continue
            l = layers.get(name)
            if name.endswith('_act'):
                node('Sigmoid', ins, name+'_sigmoid')
                node('Mul', [ins[0],name+'_sigmoid'], name)
            elif l is not None and hasattr(l, 'Weights'):
                kh,kw = map(int,l.FilterSize)
                ci,co = int(l.NumChannels),int(l.NumFilters)
                weights = np.asarray(l.Weights.Value).reshape(kh,kw,ci,co)
                weights = weights.transpose(3,2,0,1).copy()
                bias = np.asarray(l.Bias.Value, np.float32).reshape(co)
                wt,bt = constant(name+'_W',weights),constant(name+'_B',bias)
                pads = list(map(int,l.PaddingSize))
                node('Conv',ins+[wt,bt],name,kernel_shape=[kh,kw],
                     strides=list(map(int,l.Stride)),pads=[pads[0],pads[2],pads[1],pads[3]],
                     dilations=list(map(int,l.DilationFactor)))
            elif l is not None and hasattr(l,'TrainedMean'):
                args = [constant(name+'_'+k,np.asarray(a,np.float32).reshape(-1)) for k,a in
                        [('scale',l.Scale.Value),('offset',l.Offset.Value),('mean',l.TrainedMean),('var',l.TrainedVariance)]]
                node('BatchNormalization',ins+args,name,epsilon=float(l.Epsilon))
            elif l is not None and hasattr(l,'PoolSize'):
                pads=list(map(int,l.PaddingSize))
                node('MaxPool',ins,name,kernel_shape=list(map(int,l.PoolSize)),
                     strides=list(map(int,l.Stride)),pads=[pads[0],pads[2],pads[1],pads[3]])
            elif name.startswith('Cat'):
                node('Concat',ins,name,axis=1)
            elif name.startswith('addition'):
                node('Add',ins,name)
            elif name in ('layer','layer_1'):
                roi=constant(name+'_roi',np.array([],np.float32))
                scales=constant(name+'_scales',np.array([1,1,2,2],np.float32))
                node('Resize',ins[:1]+[roi,scales],name,mode='nearest',coordinate_transformation_mode='asymmetric',nearest_mode='floor')
            else:
                raise ValueError(f'Unsupported layer: {name}')
            done.add(name); advanced = True
        if not advanced: raise ValueError(f'Unresolved graph: {needed-done}')
    outputs=[H.make_tensor_value_info(n,T.FLOAT,[1,9,s,s]) for n,s in zip(targets,[52,26,13])]
    graph=H.make_graph(nodes,'Official MATLAB YOLOX cone detector',
                       [H.make_tensor_value_info('images',T.FLOAT,[1,3,416,416])],outputs,initializers)
    model=H.make_model(graph,opset_imports=[H.make_opsetid('',13)],producer_name='official-mat-cone-extractor')
    model.ir_version=8
    model=onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model,full_check=True)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,str(destination))
    record={'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'onnx_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'classes':classes,'input':'RGB float32 NCHW, 0..255, 416x416',
            'outputs':'raw NCHW [tx,ty,tw,th,objectness,4 class logits], strides 8/16/32',
            'nodes':len(nodes),'train_performed':False,
            'limitation':'No MATLAB runtime available for direct MATLAB/ONNX numerical parity.'}
    destination.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();export(a.source,a.output)
