import csv,hashlib,inspect,json,os,random
from pathlib import Path
import numpy as np
import torch

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,Path):return str(value)
    return value

def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,p)

def append_csv(path,row):
    path=Path(path);exists=path.exists()
    with path.open('a',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row))
        if not exists:writer.writeheader()
        writer.writerow({k:json.dumps(clean(v),separators=(',',':')) if isinstance(v,(list,dict,tuple,np.ndarray)) else v for k,v in row.items()})

def source_manifest():
    root=Path(__file__).resolve().parents[1]
    # Tunable JSON is saved in each run/checkpoint and checked separately.
    # Editing another recipe must not invalidate an unrelated trained model.
    files=sorted((root/'priority_mec').rglob('*.py'))
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {'sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),'files':hashes}

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
        'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])

def save_checkpoint(path,state):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');torch.save(state,temp);os.replace(temp,path)

def load_checkpoint(path):
    kw={'map_location':'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:kw['weights_only']=False
    return torch.load(path,**kw)
