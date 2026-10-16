"""External evidence persistence, causal retrieval, and versioned action queue."""
from collections import deque
import hashlib
import json
import re
from pathlib import Path
import urllib.request


def call_service(port,request,timeout=180):
    data=json.dumps(request).encode()
    req=urllib.request.Request(f'http://127.0.0.1:{port}/infer',data=data,
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.load(response)


def parse_json(text):
    text=text.strip()
    if text.startswith('```'):
        text=text.split('\n',1)[1].rsplit('```',1)[0]
    value=json.loads(text)
    if not isinstance(value,dict):raise ValueError('Expected JSON object')
    return value


class EvidenceStore:
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.manifest=self.folder/'manifest.json'
        self.records=json.loads(self.manifest.read_text()) if self.manifest.exists() else []
    def add(self,frame_id,camera,image,source='observation',note=''):
        name=f'{frame_id:06d}_{camera}_{len(self.records):06d}.png'
        dest=self.folder/name;image.save(dest)
        row={'frame_id':int(frame_id),'camera':camera,'path':name,'source':source,
             'note':note,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
        self.records.append(row);self.manifest.write_text(json.dumps(self.records,indent=2))
        return row
    def retrieve(self,frame_id,camera,as_of,bbox=None):
        if not isinstance(frame_id,int) or frame_id>as_of:raise ValueError('Future evidence rejected')
        matches=[r for r in self.records if r['frame_id']==frame_id and r['camera']==camera and r['source']=='observation']
        if not matches:raise ValueError('No real observation at requested frame')
        row=matches[-1];path=self.folder/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Evidence integrity failure')
        if bbox is not None:
            from PIL import Image
            im=Image.open(path);x1,y1,x2,y2=bbox
            if not (0<=x1<x2<=im.width and 0<=y1<y2<=im.height):raise ValueError('Crop outside image')
            path=self.folder/f'inspect_{frame_id}_{camera}_{x1}_{y1}_{x2}_{y2}.png'
            im.crop(bbox).save(path)
        return path
    def visible_manifest(self,as_of):
        return [{k:r[k] for k in ['frame_id','camera','source','note']}
                for r in self.records if r['frame_id']<=as_of and r['source']=='observation']
    def save_state(self,state):
        (self.folder/'working.json').write_text(json.dumps(state,indent=2))
    def load_state(self):return json.loads((self.folder/'working.json').read_text())


class ActionQueue:
    def __init__(self):
        self.version=0;self.subgoal=None;self.pending=deque();self.cancelled=0
    def set_subgoal(self,subgoal):
        if subgoal!=self.subgoal:
            self.cancelled+=len(self.pending);self.pending.clear()
            self.subgoal=subgoal;self.version+=1
    def fill(self,actions):
        for action in actions:self.pending.append((self.version,action))
    def pop(self):
        version,action=self.pending.popleft()
        if version!=self.version:raise ValueError('Stale action rejected')
        return action


def validate_plan(plan,as_of):
    for name in ['left_number','middle_number']:
        value=plan.get(name)
        if value is not None and (type(value)!=int or not 1<=value<=9):
            raise ValueError('Invalid target number')
    if plan.get('phase') not in ['left','middle','confirm','unknown']:
        raise ValueError('Invalid phase')
    refs=plan.get('evidence_frame_ids',[])
    if not isinstance(refs,list) or any(type(x)!=int or x<0 or x>as_of for x in refs):
        raise ValueError('Invalid evidence time')
    return plan


def instruction_for_phase(phase):
    return {'left':'Press the left button once.',
            'middle':'Press the middle button once.',
            'confirm':'Press the right confirmation button once.'}[phase]


def reconcile_counter_plan(plan,counts,as_of):
    """Enforce known task counting rules; never invent unreadable target numbers."""
    validate_plan(plan,as_of)
    for name in ['left','middle','confirm']:
        if type(counts.get(name))!=int or counts[name]<0:
            raise ValueError('Confirmed counters must be nonnegative integers')
    left=plan.get('left_number');middle=plan.get('middle_number')
    if left is None or middle is None:
        phase='unknown';reason='missing_visual_target_numbers'
    elif counts['left']>left or counts['middle']>middle:
        phase='unknown';reason='overpress_requires_recovery'
    elif counts['left']<left:
        phase='left';reason='left_count_below_target'
    elif counts['middle']<middle:
        phase='middle';reason='middle_count_below_target'
    elif counts['confirm']==0:
        phase='confirm';reason='both_target_counts_met'
    else:
        phase='unknown';reason='confirmation_already_attempted_check_environment_success'
    return {'phase':phase,'model_phase':plan['phase'],'model_phase_consistent':phase==plan['phase'],
            'rule':reason,'source':'deterministic_counter_invariant_NOT_new_VLM_inference'}


def normalize_evidence_plan(plan,as_of,available_frame_ids):
    """Normalize digit-string frame IDs, then require existing real evidence."""
    normalized=dict(plan)
    refs=plan.get('evidence_frame_ids',[])
    if not isinstance(refs,list):raise ValueError('Evidence IDs must be a list')
    normalized['evidence_frame_ids']=[int(x) if isinstance(x,str) and re.fullmatch(r'[0-9]+',x) else x for x in refs]
    validate_plan(normalized,as_of)
    if any(x not in available_frame_ids for x in normalized['evidence_frame_ids']):
        raise ValueError('Referenced frame is not an available real observation')
    if any(normalized.get(k) is not None for k in ['left_number','middle_number']) and not normalized['evidence_frame_ids']:
        raise ValueError('Visual target numbers require an evidence reference')
    return normalized
