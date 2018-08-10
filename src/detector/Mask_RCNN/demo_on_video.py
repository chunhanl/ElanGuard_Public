# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.POST_NMS_ROIS_INFERENCE = 50
config.POST_NMS_ROIS_TRAINING  = 50
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']








# -*- coding: utf-8 -*-
import cv2
from skimage.measure import find_contours
from time import time
test_img = cv2.imread('./images/7581246086_cf7bbb7255_z.jpg')
print('Warming up detector MaskRCNN....')
s = time()
model.detect([test_img], verbose=0)
e = time()
print('MaskRCNN Warmed, Time: {}'.format(e-s))



def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    colors = []
    for _ in range(N):
        rnd_hsv = np.uint8([[np.random.randint(low=0,high=255,size=3)]])
        rnd_rgb = cv2.cvtColor( rnd_hsv,cv2.COLOR_HSV2RGB)
        colors.append((rnd_rgb[0,0,0],rnd_rgb[0,0,1],rnd_rgb[0,0,2]))
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] ,
                                  image[:, :, c])
    return image
    

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union



TEST_VIDEO_PATH = './video/Cam6.avi'

cap = cv2.VideoCapture(0)
cap.open(TEST_VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES,9050)

cv2.namedWindow('frame',cv2.WINDOW_GUI_EXPANDED)
cv2.resizeWindow('frame', 1200,800)



sigma_l = 0
sigma_h = 0.5
sigma_iou = 0.5
t_min = 2

file_id = 0
tracks_active = []
tracks_finished = []


    

 
show_mask=True
show_bbox=False
captions=None


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    start_time = time()
    
    # Run detection
    results = model.detect([frame], verbose=0)
    detect_time = time()
    print("Detection Time:",detect_time-start_time)
    
    # Visualize results
    r = results[0]



    # Number of instances
    N = r['rois'].shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert r['rois'].shape[0] == r['masks'].shape[-1] == r['class_ids'].shape[0]



    # ------------------------------iou tracker----------------------------------------
    dets =[]
    for i in range(N):
        if r['class_ids'][i] <= 1:
            dets.append({'bbox': r['rois'][i], 'score': r['scores'][i], 'class_id':r['class_ids'][i], 'mask':r['masks'][:,:,i]})
    dets = [det for det in dets if det['score'] >= sigma_l]

    updated_tracks = []
    for track in tracks_active:
        track_is_updated = False
        
        if len(dets) > 0:
            # get det with highest iou
            best_match = max(dets, key=lambda x: iou(track['bboxes'][-1], x['bbox']))
            if iou(track['bboxes'][-1], best_match['bbox']) >= sigma_iou:
                track['bboxes'].append(best_match['bbox'])
                track['maskes'].append(best_match['mask'])
                track['scores'].append(best_match['score'])
                track['frames'].append(frame)
                track['frame_num'].append(frame_num)
                track['max_score'] = max(track['max_score'], best_match['score'])
                updated_tracks.append(track)
                track_is_updated = True
                # remove from best matching detection from detections
                for idx, d in enumerate(dets):
                    if np.all(best_match['bbox']==d['bbox']):
                        del dets[idx]
                        break


    # create new tracks
    new_tracks = [{'bboxes': [det['bbox']], 'maskes':[det['mask']], 'scores': [det['score']], 'frames': [frame], 'frame_num': [frame_num], 'max_score': det['score'], \
                   'class_id':det['class_id'], 'color':np.random.randint(low=0,high=255,size=3) } for det in dets]
    tracks_active = updated_tracks + new_tracks

    track_time = time()
    print("Tracking Time:",track_time-detect_time)
    # ------------------------------iou tracker----------------------------------------
    

    # Show area outside image boundaries.
    height, width = frame.shape[:2]


    masked_image = frame.astype(np.uint32).copy()
    for i,track in enumerate(tracks_active):
        color = (int(track['color'][0]),int(track['color'][1]),int(track['color'][2]))

        # Bounding box
        if not np.any(track['bboxes'][-1]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = track['bboxes'][-1]
        if show_bbox:
            cv2.rectangle(frame,(x1, y1),(x2, y2),color,3)

        # Label
        if not captions:
            class_id = track['class_id']
            score = track['max_score'] 
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
            

        cv2.putText(frame, caption, (x1,y1+8), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.9, color=(255,255,255))
        
        
        # Mask
        mask = track['maskes'][-1]
        if show_mask:
            frame = apply_mask(frame, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            verts = np.int32(verts.reshape((-1,1,2)))
            cv2.fillPoly(frame, verts, color=color)


    draw_time = time()
    print("Drawing Time:",draw_time-track_time)
    
    fps = 1.0/ (detect_time-start_time )
    cv2.putText(frame, str("%.2f" % fps), (1800,50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=(255,255,255))
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()







