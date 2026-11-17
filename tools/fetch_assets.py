"""Download original public MathWorks weights and test clip with SHA-256 checks."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

BASE='https://raw.githubusercontent.com/matlab-deep-learning/Cone-Detection-Using-Deep-Learning-for-Formula-Student-Driverless/main/'
ASSETS={
    'yoloxConeDetector.mat':'fcd732b620e52b8110e6c3ce869af86dc7d91a18abac2113bbee391d859275bd',
    'videos/testConeDetector_416x416.mp4':'c62e834eee82666702afa0969fca70a424f3c520eee3adbb4363cd12fe6f85ef',
}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='assets');a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    for remote,expected in ASSETS.items():
        target=root/Path(remote).name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:
            print('Already verified:',target);continue
        temp=target.with_suffix(target.suffix+'.part')
        try:
            with urlopen(BASE+remote,timeout=60) as response,temp.open('wb') as out:
                while chunk:=response.read(1024*1024):out.write(chunk)
            if hashlib.sha256(temp.read_bytes()).hexdigest()!=expected:raise ValueError('Checksum mismatch: '+remote)
            temp.replace(target);print('Verified:',target)
        finally:temp.unlink(missing_ok=True)


if __name__=='__main__':main()
