"""Download and verify only the HyperKvasir cecum class from official OSF storage."""
import argparse,hashlib,json,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/hyperkvasir_cecum'
ENDPOINT='https://api.osf.io/v2/nodes/mh9sj/files/osfstorage/5dfb5b530236b8000c746a8e/?page[size]=100'

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def get_json(url,retries=5):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url,timeout=90) as r:return json.load(r)
        except Exception:
            if attempt+1==retries:raise
            time.sleep(2**attempt)

def inventory():
    records=[];url=ENDPOINT
    while url:
        page=get_json(url);records.extend(page['data']);url=page['links']['next']
    items=[]
    for r in records:
        a=r['attributes'];items.append({'name':a['name'],'bytes':int(a['size']),
          'sha256':a['extra']['hashes']['sha256'],'download':r['links']['download'],'osf_id':r['id']})
    items.sort(key=lambda x:x['name'])
    if len(items)!=1009 or len({x['name'] for x in items})!=1009:raise ValueError(f'Expected 1009 unique files, got {len(items)}')
    return items

def download(item,retries=5):
    DATA.mkdir(parents=True,exist_ok=True);target=DATA/item['name'];part=target.with_suffix(target.suffix+'.part')
    if target.is_file() and target.stat().st_size==item['bytes'] and sha256(target)==item['sha256']:return target
    if part.exists():part.unlink()
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(item['download'],timeout=120) as src,part.open('wb') as dst:
                while True:
                    block=src.read(1024*1024)
                    if not block:break
                    dst.write(block)
            if part.stat().st_size!=item['bytes'] or sha256(part)!=item['sha256']:raise ValueError(f'Hash/size mismatch: {item["name"]}')
            part.replace(target);return target
        except Exception:
            if part.exists():part.unlink()
            if attempt+1==retries:raise
            time.sleep(2**attempt)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inventory-only',action='store_true')
    ap.add_argument('--workers',type=int,default=12);a=ap.parse_args()
    items=inventory();DATA.mkdir(parents=True,exist_ok=True)
    if not a.inventory_only:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(download,item):item for item in items}
            for i,future in enumerate(as_completed(futures),1):
                item=futures[future];future.result()
                print(json.dumps({'verified':i,'total':len(items),'name':item['name']}),flush=True)
    manifest={'dataset':'HyperKvasir','class':'cecum','source_endpoint':ENDPOINT,
      'license':'CC BY 4.0','files':items,'total_bytes':sum(x['bytes'] for x in items)}
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(items),'bytes':manifest['total_bytes'],'manifest':str(DATA/'manifest.json')}))
if __name__=='__main__':main()
