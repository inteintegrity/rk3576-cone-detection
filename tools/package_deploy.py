"""Create a board deployment bundle outside the clean GitHub source tree."""
import argparse
from pathlib import Path
import tarfile

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[1]
target=Path(a.output).resolve();target.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(target,'w:gz') as tar:
    for path in sorted(root.rglob('*')):
        rel=path.relative_to(root)
        if path.is_file() and path.name!='.env' and not any(x in {'.git','__pycache__','.pytest_cache','.venv','.deps','.ffmpeg','data','assets','models','outputs','validation'} for x in rel.parts):
            tar.add(path,arcname=str(rel))
    tar.add(a.model,arcname='models/cone_yolox_rk3576_fp16.rknn')
print(target)
