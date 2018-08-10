# -*- coding: utf-8 -*-
import numpy as np

class Tracklet():
    def __init__(self,):

        self.trackelt_num = None
        
        self.elan_id = 'unidentified'     #'unidentified'/'unknown'/ xxxxx(known) 
        self.color   =  None
        self.cam_id  =  None
        self.active  =  None
        
        self.person_imgs         =  []
        self.person_bboxs        =  []
        self.person_embeds       =  []
        self.person_embeds_mean  =  None

        self.face_imgs           =  []
        self.face_embeds         =  []
        self.face_embeds_mean    =  None
        
        
class Tracklets():
    def __init__(self, key=None, value=None):  
        self.dict = {}  
        if key is not None:
            self.dict[key] = value  
        
    def __getitem__(self, key=None):  
        return self.dict[key]  
    
    def __setitem__(self, key=None, value=None):  
        self.dict[key] = value  
        
    def __len__(self):  
        return len(self.dict)  
    
    def keys(self):
        return self.dict.keys()
    
    def items(self):
        return self.dict.items()
        
    def get_tracklet(self, tracklet_num):
        return self.dict[tracklet_num] if tracklet_num in self.dict.keys() else print('Can not find tracklet num: {}'.format(tracklet_num))

                   
    def add_tracklet(self, det):
        new_tracklet = Tracklet()   
        new_tracklet.tracklet_num       = det['s_tracklet_num']
        new_tracklet.elan_id            = det['s_elan_id']
        new_tracklet.cam_id             = det['e_cam_id']
        new_tracklet.person_bboxs       = [det['person_bbox']]   if det['person_bbox']   is not None else []
        new_tracklet.person_embeds      = [det['person_embed']]  if det['person_embed']  is not None else []
        new_tracklet.person_embeds_mean = det['person_embed']    if det['person_embed']  is not None else None
        new_tracklet.face_embeds        = [det['face_embed']]    if det['face_embed']    is not None else []
        new_tracklet.face_embeds_mean   = det['face_embed']      if det['face_embed']    is not None else None
        #For Visualization Only
        new_tracklet.color              = det['s_tracklet_color']
        new_tracklet.person_imgs        = [det['person_img']]   if det['person_img']   is not None else []
        new_tracklet.face_imgs          = [det['face_img']]     if det['face_img']     is not None else []
        self.dict[det['s_tracklet_num']]= new_tracklet

    def update_tracklet(self, det):
        match_tracklet = self.get_tracklet(det['s_tracklet_num'])      
        #Clear out poor faces
        if (det['s_elan_id']!='unidentified' and det['s_elan_id']!='unknown'):
            if det['face_embed'] is not None and match_tracklet.face_embeds_mean is not None:
               similarity = np.dot( det['face_embed'],match_tracklet.face_embeds_mean.T )                      
               if similarity <0.4:   
                   det['face_bbox'] = None
                   det['face_img'] = None
                   det['face_keypoint'] = None
                   det['face_embed'] = None
                   det['face_confidence'] = None
        #For Visualization Only
        match_tracklet.person_imgs.append(  det['person_img']  )
        
        match_tracklet.person_bboxs.append( det['person_bbox'] )
        match_tracklet.person_embeds.append(det['person_embed'])
        match_tracklet.person_embeds_mean = det['person_embed'] if  match_tracklet.person_embeds_mean is None  \
                                            else (match_tracklet.person_embeds_mean* (len(match_tracklet.person_embeds)-1) + det['person_embed'])/len(match_tracklet.person_embeds)
        if det['face_embed'] is not None:
            #For Visualization Only
            match_tracklet.face_imgs.append(  det['face_img']  )
        
            match_tracklet.face_embeds.append(det['face_embed'])
            match_tracklet.face_embeds_mean = det['face_embed'] if match_tracklet.face_embeds_mean is None  \
                                            else (match_tracklet.face_embeds_mean* (len(match_tracklet.face_embeds)-1) + det['face_embed'])/len(match_tracklet.face_embeds)


        
        