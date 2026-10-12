"""Fetch the eight Amazon 2014 files from the original Stanford source."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.request

BASE = Path('/root/genrec-reproduction')
RAW = BASE / 'genrec/dataset/amazon/raw'
CATEGORIES = {'beauty':'Beauty', 'sports':'Sports_and_Outdoors',
              'toys':'Toys_and_Games', 'home':'Home_and_Kitchen'}
SOURCE = 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles'

def ranged_download(url, destination):
    head=urllib.request.Request(url,method='HEAD')
    with urllib.request.urlopen(head,timeout=60) as response:
        expected=int(response.headers['Content-Length'])
    part_dir=destination.with_suffix(destination.suffix+'.parts')
    part_dir.mkdir(exist_ok=True)
    chunk_size=4*1024*1024
    ranges=[(index,start,min(start+chunk_size,expected)-1)
            for index,start in enumerate(range(0,expected,chunk_size))]
    def piece(spec):
        index,start,end=spec
        path=part_dir/f'{index:05d}'
        if path.exists() and path.stat().st_size==end-start+1:
            return path
        for attempt in range(5):
            try:
                request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=60) as response,path.open('wb') as out:
                    if response.status!=206:
                        raise IOError('Source did not honor byte range')
                    expected_range=f'bytes {start}-{end}/{expected}'
                    if response.headers.get('Content-Range')!=expected_range:
                        raise IOError('Source returned a different byte range')
                    while data:=response.read(512*1024):
                        out.write(data)
                if path.stat().st_size!=end-start+1:
                    raise IOError('Truncated range')
                print(f'{destination.name}: chunk {index+1}/{len(ranges)} verified',flush=True)
                return path
            except Exception as error:
                if attempt==4:
                    raise
                time.sleep(3*(attempt+1))
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths=list(pool.map(piece,ranges))
    with destination.open('wb') as out:
        for path in paths:
            with path.open('rb') as inp:
                while data:=inp.read(1024*1024):
                    out.write(data)
    assert destination.stat().st_size==expected

def fetch(task):
    split, filename = task
    dest = RAW / split / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + '.part')
    if not dest.exists():
        for attempt in range(5):
            try:
                ranged_download(f'{SOURCE}/{filename}',temporary)
                with gzip.open(temporary,'rb') as inp:
                    while inp.read(4*1024*1024):
                        pass
                temporary.replace(dest)
                break
            except Exception as error:
                print(f'{filename}, attempt {attempt+1}: {error}', flush=True)
                if attempt == 4:
                    raise
                time.sleep(5*(attempt+1))
    digest = hashlib.sha256()
    with dest.open('rb') as inp:
        while chunk := inp.read(4*1024*1024):
            digest.update(chunk)
    record = {'split':split,'filename':filename,'bytes':dest.stat().st_size,
              'sha256':digest.hexdigest(),'source':f'{SOURCE}/{filename}'}
    with gzip.open(dest,'rb') as inp:
        while inp.read(4*1024*1024):
            pass
    dest.with_suffix(dest.suffix+'.verified.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record), flush=True)
    return record

if __name__ == '__main__':
    tasks = [(split, f'{prefix}{name}{suffix}.json.gz')
             for split,name in CATEGORIES.items()
             for prefix,suffix in [('reviews_','_5'),('meta_','')]]
    status={'state':'downloading'}
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(fetch,tasks))
        status = {'state':'ready','files':records}
    except Exception as error:
        status = {'state':'failed','error':str(error)}
        raise
    finally:
        manifest = BASE / 'manifests/data-status.json'
        manifest.parent.mkdir(parents=True,exist_ok=True)
        manifest.write_text(json.dumps(status,indent=2))
