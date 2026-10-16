"""Persistent visual-instance contract for observe-and-pick development checks.

Inputs: VLM-selected category, real image boxes and evidence frame IDs. Outputs:
a stable external target identity and validated boxes. No simulator actor IDs,
ground-truth locations, embeddings or learned parameters are introduced.
"""
import json
from pathlib import Path

CATEGORIES={'car','mouse','stapler','bell','cards','unknown'}


def bind_input_evidence(plan,supplied_frame_ids):
    """Bind missing provenance to actual request images, never inventing frames.

    Raw responses must be preserved separately. Model-supplied references are
    validated, not repaired; missing fields use known request provenance and
    explicitly identify the controller as the source of the references.
    """
    result=dict(plan)
    if 'evidence_frame_ids' not in result:
        result['evidence_frame_ids']=sorted(set(supplied_frame_ids))
        result['evidence_binding']='controller_bound_actual_input_frames_NOT_model_citation'
    else:
        refs=result['evidence_frame_ids']
        if not isinstance(refs,list) or not refs or any(type(v)!=int or v not in supplied_frame_ids for v in refs):
            raise ValueError('Model cited unavailable input evidence')
        result['evidence_binding']='model_citation_validated_against_inputs'
    return result


def validate_box(box):
    """Validate an integer 0–1000 box with positive area, rejecting booleans."""
    if not isinstance(box,list) or len(box)!=4 or any(type(v)!=int for v in box):
        raise ValueError('Box requires four integer normalized coordinates')
    x1,y1,x2,y2=box
    if not (0<=x1<x2<=1000 and 0<=y1<y2<=1000):raise ValueError('Invalid normalized box')
    return box


def pixel_box(box,size):
    """Convert a validated normalized box to a bounded pixel crop."""
    validate_box(box);width,height=size
    x1,y1,x2,y2=box
    result=[int(x1*width/1000),int(y1*height/1000),int(x2*width/1000),int(y2*height/1000)]
    if result[0]>=result[2] or result[1]>=result[3]:raise ValueError('Degenerate pixel crop')
    return result


def persist_target(path,plan,target_id,as_of):
    """Store visual target identity with real causal evidence; return the state.

    Frame IDs must be available by as_of. Unknown plans are preserved, never
    completed with truth labels. target_id is an external label, not an actor ID.
    """
    category=plan.get('category')
    if category not in CATEGORIES:raise ValueError('Invalid visual category')
    refs=plan.get('evidence_frame_ids',[])
    if not refs or any(type(v)!=int or v<0 or v>as_of for v in refs):raise ValueError('Invalid causal evidence')
    state={'target_id':target_id,'category':category,'description':str(plan.get('description','')),
           'target_bbox':None if category=='unknown' else validate_box(plan['target_bbox']),
           'reference_bbox':None if category=='unknown' else validate_box(plan['reference_bbox']),
           'evidence_frame_ids':refs,'target_version':1,'source':'VLM_real_RGB_NOT_simulator_truth'}
    state['evidence_binding']=plan.get('evidence_binding','model_citation')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(state,indent=2));return state
