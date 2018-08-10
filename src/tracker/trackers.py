# -*- coding: utf-8 -*-
"""
Tracking Input format:
input_detections = [ detection, detection ...]
original_track = [{'detection':,'result_id':}]

Tracking Output format:
[{'detection':,'result_id':}]


"""

import abc
class Tracker(abc.ABC):
    
    @abc.abstractmethod
    def track(self):
        'Return track result'
        return NotImplemented
    
import numpy as np
import cv2


'''
Tracker: IOUTracker
Version: 1.0
Maintainer: C.H.Lu
'''
class IOUTracker(Tracker):
    def iou(self, bbox1, bbox2):
        
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
    
    def __init__(self,sigma_iou=0.5):
        self.sigma_iou = sigma_iou
    def track(self, input_detections, previous_detections, IOU_found_id=None):
        
        self.found_id = [] if IOU_found_id is None else IOU_found_id
        # First track with IOU Tracker
        for det in previous_detections[-1]:           
            if len(input_detections) > 0:
                # tracking with finding the highest iou
                best_match = max(input_detections, key=lambda x: self.iou(det['person_bbox'], x['person_bbox']))
                if self.iou(det['person_bbox'], best_match['person_bbox']) >= self.sigma_iou:   
                    
                    # Doublecheck by PersonReID if avaliable, prevent false linking by IOUTracker
                    if det['person_embed'] is not None and best_match['person_embed'] is not None:
                        l2_distance = np.sqrt( np.sum( np.square( det['person_embed']- best_match['person_embed']))+1e-12 )
                        if l2_distance>10:
                            continue
                        
                    self.found_id.append(det['s_tracklet_num'])
                    best_match['s_tracklet_num'] = det['s_tracklet_num']     
                    best_match['s_tracklet_color'] = det['s_tracklet_color']
                    #----visualize tracked decided by tracker------
#                    img = best_match['person_img'].copy()
#                    h,w,_ = img.shape
#                    cv2.rectangle(img,(0, 0),(w,h),(255,0,0),5)
#                    best_match['person_img'] =img

        return input_detections



'''
Tracker: TripletTracker
Version: 1.0
Maintainer: C.H.Lu
'''
class TripletTracker(Tracker):
   
    def __init__(self,):
        self.link_distance = 10
        self.move_ratio_limit = 0.6
        self.iou_tracker = IOUTracker()
        pass
    
    def track(self, input_detections, previous_detections, IOU_found_id=None):
        self.found_id = [] if IOU_found_id is None else IOU_found_id
        # Track with feature linking
        for in_det in input_detections:  
            if in_det['s_tracklet_num'] ==-1:
                for idx in range(1,len(previous_detections)):    
                    min_distance=100
                    min_distance_det= None
                    for p_det in previous_detections[-idx]:
                        if  p_det['person_embed'] is not None:
                            l2_distance = np.sqrt( np.sum( np.square( p_det['person_embed']- in_det['person_embed']))+1e-12 )
                            if l2_distance<min_distance:
                                min_distance_det = p_det  
                                min_distance = l2_distance
                    
                    if min_distance_det is not None:
                        if  ((min_distance <= self.link_distance ) and (min_distance_det['s_tracklet_num'] not in self.found_id)):
                            x1, y1, x2, y2 =  min_distance_det['person_bbox']
                            midx, midy = (x1+x2)//2, (y1+y2)//2
                            inx1, iny1, inx2, iny2 =  in_det['person_bbox']
                            in_midx, in_midy = (inx1+inx2)//2, (iny1+iny2)//2
                            physical_limit = (((midx-in_midx)/(x2-x1))**2 + ((midy-in_midy)/(y2-y1))**2)**0.5
                            if( physical_limit <= self.move_ratio_limit):
                                if self.iou_tracker.iou(in_det['person_bbox'], min_distance_det['person_bbox'])>= self.iou_tracker.sigma_iou:   
                                    in_det['s_tracklet_num']  =  min_distance_det['s_tracklet_num']
                                    in_det['s_tracklet_color']=  min_distance_det['s_tracklet_color']
                                    self.found_id.append(min_distance_det['s_tracklet_num'])
                                    #----visualize tracked decided by link------
#                                    img = in_det['person_img'].copy()
#                                    h,w,_ = img.shape
#                                    cv2.rectangle(img,(0, 0),(w,h),(0,255,0),5)
#                                    in_det['person_img'] =img
                                    break
                                
        return input_detections



'''
Tracker: Triplet_w_IOU
Version: 1.0
Maintainer: C.H.Lu
'''
class Triplet_w_IOUTracker(Tracker):

    def __init__(self,):
        self.iou_tracker     = IOUTracker()
        self.triplet_tracker = TripletTracker()
        
    def track(self, input_detections, previous_detections, IOU_found_id =None):
        
        self.found_id = [] if IOU_found_id is None else IOU_found_id
        # First track with IOU Tracker
        input_detections = self.iou_tracker.track(input_detections, previous_detections, self.found_id)
        # Second track with feature linking
        input_detections = self.triplet_tracker.track(input_detections, previous_detections, self.found_id)
         
        return input_detections
    
    
    
'''
Tracker: AlignReidTracker
Version: 1.0
Maintainer: C.H.Lu
'''
class AlignReidTracker(Tracker):
   
    def __init__(self,):
        self.link_distance = 0.85
        self.move_ratio_limit = 0.6
        self.iou_tracker = IOUTracker()
        pass
    
    def track(self, input_detections, previous_detections, IOU_found_id=None):
        self.found_id = [] if IOU_found_id is None else IOU_found_id
        # Track with feature linking
        for in_det in input_detections:  
            if in_det['s_tracklet_num'] ==-1:
                for idx in range(1,len(previous_detections)):    
                    min_distance=100
                    min_distance_det= None
                    for p_det in previous_detections[-idx]:
                        if  p_det['person_embed'] is not None:
                            l2_distance = np.sqrt( np.sum( np.square( p_det['person_embed']- in_det['person_embed']))+1e-12 )
                            if l2_distance<min_distance:
                                min_distance_det = p_det  
                                min_distance = l2_distance
                    
                    if min_distance_det is not None:
                        if  ((min_distance <= self.link_distance ) and (min_distance_det['s_tracklet_num'] not in self.found_id)):
                            x1, y1, x2, y2 =  min_distance_det['person_bbox']
                            inx1, iny1, inx2, iny2 =  in_det['person_bbox']
                            physical_limit = (((x1-inx1)/(x2-x1))**2 + ((y1-iny1)/(y2-y1))**2)**0.5
                            if( physical_limit <= self.move_ratio_limit):
                                if self.iou_tracker.iou(in_det['person_bbox'], min_distance_det['person_bbox'])>= self.iou_tracker.sigma_iou:   
                                    in_det['s_tracklet_num']  =  min_distance_det['s_tracklet_num']
                                    in_det['s_tracklet_color']=  min_distance_det['s_tracklet_color']
                                    self.found_id.append(min_distance_det['s_tracklet_num'])
                                    #----visualize tracked decided by link------
                                    img = in_det['person_img'].copy()
                                    h,w,_ = img.shape
                                    cv2.rectangle(img,(0, 0),(w,h),(0,255,0),5)
                                    in_det['person_img'] =img
                                    break
                                
        return input_detections
    
'''
Tracker: AlignReid_w_IOU
Version: 1.0
Maintainer: C.H.Lu
'''
class AlignReid_w_IOUTracker(Tracker):

    def __init__(self,):
        self.iou_tracker     = IOUTracker()
        self.triplet_tracker = AlignReidTracker()
        
    def track(self, input_detections, previous_detections, IOU_found_id =None):
        
        self.found_id = [] if IOU_found_id is None else IOU_found_id
        # First track with IOU Tracker
        input_detections = self.iou_tracker.track(input_detections, previous_detections, self.found_id)
        # Second track with feature linking
        input_detections = self.triplet_tracker.track(input_detections, previous_detections, self.found_id)
         
        return input_detections