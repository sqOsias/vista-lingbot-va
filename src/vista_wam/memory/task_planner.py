"""Nine-task visual planner with explicitly ablated persistent evidence.

Inputs: fixed public task goal, current real RGB, optional past real frames and
actual execution attempts. Outputs: one validated primitive proposal and raw
VLM receipt. Current-only requests receive neither past images nor event logs;
model interpretation is labeled separately from real execution evidence.
"""
import json
from pathlib import Path
from PIL import Image
from vista_wam.core.evidence import call_service,parse_json
from vista_wam.memory.target import validate_box

TASK_GOALS={
 'observe_and_pickup':'Pick up the table instance matching the earlier shelf reference after it becomes occluded.',
 'put_back_block':'Remember the initial red block mat. Move the red block to the center between the four blue mats, press the button once, then return the block to its ORIGINAL mat and release.',
 'rearrange_blocks':'Move the initially center red block to the initially empty blue mat. Press the button exactly once. Move the other red block from its original mat to the center between the two mats, then release.',
 'swap_blocks':'Exchange the ORIGINAL positions of the two red blocks in three trays. Use the initially empty tray as a buffer, do not stack blocks. When exchanged, press the confirmation button exactly once.',
 'swap_T':'Swap the two colored T blocks positions AND their original planar orientations. Use a free table region as a buffer. Release both grippers when finished.',
 'press_button':'Read the two number cards. Press the left button exactly the left-card number of times, then the middle button exactly the right-card number of times, then press the right confirmation button. Remember executed press attempts; do not assume a requested but failed motion pressed it.',
 'cover_blocks':'First cover the left, middle, then right colored cube with its corresponding lid. Turn the upright lids to lie over the cubes. Then uncover the cubes in RED, GREEN, BLUE order by moving each covering lid back to its original rear position. The colors become hidden when covered.',
 'blocks_ranking_try':'Find the hidden correct left-to-right order of the three different colored cubes by trying permutations and pressing the confirmation button. The desired color order is not given. Avoid retrying the same arrangement when history is available. Use free table space as a buffer and release the right gripper before confirming.',
 'battery_try':'Place both batteries in the two slots. Find the hidden correct pair of battery polarities by trying combinations and reading the gauge. Rotate a battery by 180 degrees around the vertical axis to reverse it. Finish with both batteries seated and the gauge on.'
}


def validate_action(value):
    """Accept bounded image-grounded primitive proposals; reject invented poses."""
    allowed=['lift','pick_place','press','push','unknown','done']
    if value.get('action') not in allowed:raise ValueError('Unsupported primitive')
    if value['action'] not in ['unknown','done']:
        validate_box(value.get('source_bbox'))
        if value['action'] in ['pick_place','push']:validate_box(value.get('destination_bbox'))
    rotations=value.get('rotation_xyz_degrees',[0,0,0])
    if not isinstance(rotations,list) or len(rotations)!=3 or any(type(v) not in [int,float] or abs(v)>180 for v in rotations):raise ValueError('Invalid requested rotation')
    if value.get('arm','auto') not in ['auto','left','right']:raise ValueError('Invalid arm')
    return value


class VisualTaskPlanner:
    """Plan one action from available evidence, with no simulator labels."""

    def __init__(self,task,condition,output):
        """Fix public task semantics and current/memory evidence visibility."""
        if task not in TASK_GOALS or condition not in ['memory','current']:raise ValueError('Unknown task/condition')
        self.task=task;self.condition=condition;self.output=Path(output);self.output.mkdir(parents=True);self.frames=[];self.events=[];self.notes=[];self.calls=0

    def record(self,path,event=None):
        """Persist a real observation and optional actual-attempt event only."""
        self.frames.append(str(Path(path).resolve()))
        if event is not None:self.events.append(dict(event,source='actual_executed_attempt_NOT_confirmation'))

    def request_evidence(self,current):
        """Expose past records only to memory; current gets a single fresh frame."""
        current=str(Path(current).resolve())
        if self.condition=='current':return [current],{'condition':'current','prior_events':[],'prior_interpretations':[]}
        candidates=self.frames[:1]+self.frames[-2:]+[current];images=list(dict.fromkeys(candidates))
        return images,{'condition':'memory','prior_events':self.events[-24:],'prior_interpretations':self.notes[-8:]}

    def decide(self,current):
        """Call frozen stateless VLM, validate one proposal and retain provenance."""
        images,context=self.request_evidence(current);labels=['real earlier frame' for _ in images];labels[-1]='CURRENT real head camera'
        if self.task=='press_button':
            image=Image.open(current);crop=image.crop((0,0,image.width,int(image.height*.65)));path=self.output/f'digits_zoom_{self.calls:03d}.png';crop.resize((image.width*2,int(image.height*.65)*2),Image.Resampling.NEAREST).save(path);images.append(str(path));labels.append('CURRENT upper-image enlarged actual pixels; boxes must still refer to full CURRENT head camera')
        prompt=('Task: '+TASK_GOALS[self.task]+'\nImages in order: '+json.dumps(labels)+'. Available evidence: '+json.dumps(context)+
                '\nChoose ONE next physically useful primitive. Return JSON only: action (lift/pick_place/press/push/unknown/done), source_bbox, destination_bbox (normalized integer [x1,y1,x2,y2], 0..1000 in full CURRENT head camera), arm (auto/left/right), rotation_xyz_degrees (relative WORLD rotations, default [0,0,0]), object_description, reason, memory_note. '+
                'For pick_place, destination_bbox may be a free table spot, mat, tray or slot; never the source itself unless rotating in place. For press, box ONLY the button cap, not a number card. '+
                'Use source lift only when the task is pickup. Do not call done just because an intermediate move finished. For trial tasks, continue new trials rather than declaring unknown only because the goal is hidden. '+
                'All boxes refer to the CURRENT full image, regardless of historical image/crop dimensions. No actor IDs, target labels, physical truth counters or expert actions are available. Mark visual uncertainty as unknown; memory_note is an interpretation, not an observation.')
        response=call_service(18765,{'images':images,'prompt':prompt,'max_new_tokens':384});file=self.output/f'call_{self.calls:03d}.json';file.write_text(json.dumps({'images':images,'request_prompt':prompt,'response':response},indent=2));self.calls+=1
        value=validate_action(parse_json(response['text']))
        if self.condition=='memory' and value.get('memory_note'):self.notes.append({'source':'VLM_interpretation_NOT_raw_observation','actual_frame':str(current),'text':str(value['memory_note'])[:320]})
        return value
