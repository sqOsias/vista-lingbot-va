"""Frozen VLM bridge for current-view and instance-memory experiments.

Inputs: real initial/current RGB and optional causal target state. Outputs:
validated visual plans, actual evidence crops, raw service receipts and a short
LingBot language goal. No simulator truth is accepted by this module.
"""
import json
from pathlib import Path
from PIL import Image
from vista_wam.core.evidence import EvidenceStore, call_service, parse_json
from vista_wam.memory.target import bind_input_evidence, persist_target, pixel_box, validate_box, CATEGORIES

GENERAL = 'Pick up the table object that matches the reference object on the shelf.'


def initial_selection(image, output):
    """Match a shelf exemplar to a separate table object using real fixed-ROI inspection."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=Image.open(image)
    if source.size!=(320,240):raise ValueError('Shelf inspection ROI requires 320x240 head RGB')
    store=EvidenceStore(output/'initial_evidence');store.add(0,'head_camera',source)
    crop=store.retrieve(0,'head_camera',as_of=0,bbox=[112,28,224,96])
    zoom=output/'reference_zoom_4x_NEAREST_real_pixels.png'
    Image.open(crop).resize((448,272),Image.Resampling.NEAREST).save(zoom)
    prompt=('Image1 enlarges the real exemplar on the elevated shelf in image2. Image2 is the full INITIAL scene. '
            'The white work surface in the LOWER HALF of image2 is the robot table, even though it looks like the floor. '
            'The exemplar and target are TWO SEPARATE physical objects with matching appearance/model, not the same physical object. '
            'Select the lower-table candidate matching the exemplar color and shape; same category alone is insufficient. '
            'Return short JSON only: category (car/mouse/stapler/bell/cards/unknown), description, '
            'reference_bbox, target_bbox (both integer [x1,y1,x2,y2] in 0 to 1000 for image2), evidence_frame_ids=[0]. '
            'If uncertain return category=unknown and null boxes. No simulator labels are available.')
    return _infer([zoom,image],prompt,[0],output/'initial_response.json')


def current_selection(image, output, initial_image=None, initial_plan=None):
    """Locate a target using current RGB alone, or explicitly supplied initial evidence.

    The current-only path accepts no historical plan or image. The memory path
    preserves actual reference/candidate crops; unknown remains unknown.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if initial_image is None:
        if initial_plan is not None:raise ValueError('Current-only condition cannot receive history')
        images=[image];refs=[1]
        prompt=('This is the CURRENT head camera. Pick up the table object matching the reference on the shelf. '
                'Identify the matching INSTANCE only if the reference is visible and sufficient. '
                'You have no earlier image or target description. If it is hidden or ambiguous, return unknown; do not guess. '
                'Return short JSON: category (car/mouse/stapler/bell/cards/unknown), description, '
                'target_bbox (integer [x1,y1,x2,y2] in 0 to 1000 or null), evidence_frame_ids=[1].')
    else:
        store=EvidenceStore(output/'evidence');store.add(0,'head_camera',Image.open(initial_image))
        store.add(1,'head_camera',Image.open(image))
        state=persist_target(output/'target_memory.json',initial_plan,'visual_target_v1',as_of=0)
        if state['category']=='unknown':
            # Unknown is exactly when the original visual evidence must remain
            # available. Do not reduce C2 to current-only after an initial failure.
            images=[initial_image,image];refs=[0,1]
            prompt=('Image1 is the real INITIAL head camera before the shelf reference was hidden; image2 is the CURRENT camera. '
                    'The earlier selection was uncertain, so inspect the actual earlier reference in image1 now. '
                    'The white lower work surface is the robot table. The shelf exemplar and table target are separate '
                    'physical objects with matching appearance/model. Select the CURRENT table candidate in image2 '
                    'matching the earlier exemplar color and shape; same category alone is insufficient. '
                    'Return JSON: category (car/mouse/stapler/bell/cards/unknown), description, '
                    'target_bbox (integer [x1,y1,x2,y2] in 0 to 1000 for image2 or null), evidence_frame_ids=[0,1]. '
                    'If still ambiguous return unknown. No simulator labels are available.')
        else:
            size=Image.open(initial_image).size
            reference=store.retrieve(0,'head_camera',as_of=1,bbox=pixel_box(state['reference_bbox'],size))
            candidate=store.retrieve(0,'head_camera',as_of=1,bbox=pixel_box(state['target_bbox'],size))
            images=[reference,candidate,image];refs=[0,1]
            prompt=('Image1 is the real reference crop; image2 is the TABLE CANDIDATE selected before occlusion; '
                    'image3 is the current full head camera after the reference was hidden. '
                    'The white lower work surface is the robot table. The shelf exemplar is a separate physical object. '
                    'Locate the same TABLE CANDIDATE instance in image3, comparing color and shape with the crops. '
                    f'Persistent visual state: {json.dumps(state)}. '
                    'Return short JSON: category (car/mouse/stapler/bell/cards/unknown), description, '
                    'target_bbox (integer [x1,y1,x2,y2] in 0 to 1000 for image3 or null), evidence_frame_ids=[0,1].')
    return _infer(images,prompt,refs,output/'current_response.json')


def _infer(images,prompt,refs,path):
    response=call_service(18765,{'images':[str(Path(p).resolve()) for p in images],
                               'prompt':prompt,'max_new_tokens':512})
    path.write_text(json.dumps({'request_images':[str(p) for p in images],
                                'request_prompt':prompt,'response':response},indent=2))
    plan=bind_input_evidence(parse_json(response['text']),refs)
    if plan.get('category') not in CATEGORIES:raise ValueError('Invalid category')
    if plan['category']!='unknown':validate_box(plan['target_bbox'])
    return plan


def language_goal(plan,size):
    """Convert a valid visual box into an existing language condition, at most 180 chars."""
    if plan['category']=='unknown':return GENERAL
    x1,y1,x2,y2=validate_box(plan['target_bbox']);width,height=size
    x=(x1+x2)*width/2000;y=(y1+y2)*height/2000
    side='left' if x<width/2 else 'right'
    # Keep exact category and pixel first; a bounded appearance hint helps distinguish same-class instances.
    description=' '.join(str(plan.get('description','')).replace('/',' ').split())[:36]
    return (f'Pick up the {plan["category"]} on the {side} side near pixel ({round(x)}, {round(y)}) '
            f'in the {width}x{height} head camera and lift it. Appearance: {description}.')[:180]
