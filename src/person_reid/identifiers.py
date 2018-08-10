# -*- coding: utf-8 -*-
"""
Identifier Input format:

dets = [ detection1, detection2, detection3 ... ]
detection = 
{
    'person_bbox' : [ x1, y1, x2, y2 ]  #absolute position, x2>x1 and y2>y1
    'person_img'  : Cropped Image
    'person_seg'  : Seg in Cropped Image
    'person_embed': 
    'person_confidence': 
        
    'face_bbox'   : [ x1, y1, x2, y2 ]  #RELATIVE position, x2>x1 and y2>y1
    'face_img'    : Cropped Image
    'face_keypoint'  : 
    'face_confidence': 
}


Identifier Output format:

[{'result_id':, detection':}]

"""


# Abstract class for detectors
import abc
class Identifier(abc.ABC):
 
    @abc.abstractmethod
    def identify(self):
        'Return Identifier result'
        return NotImplemented
    
import numpy as np   

'''
Tracker: TripletReid 
Version: 1.0
Instruction:Identified identity based on tracklets with voting technique
Maintainer: C.H.Lu
'''  
class TripletReid(Identifier):
    
    def __init__(self,):
        self.min_require_votes = 5
        
        
    def identify(self, tracklets, employees, guests=None):        
        result =[]

        for key, tracklet in tracklets.items():
            if (tracklet.elan_id=='unidentified' or 'guest' in tracklet.elan_id):
                # Reidentify if tracklet is 'unidentified' or is 'guest'
                if len(tracklet.person_embeds) >= self.min_require_votes: 
                    # Check if tracklet belongs to any employee
                    self.is_employee = False
                    self.embed_distance   = np.ones(len(employees))*1000
                    for jdx, (elan_id,identity) in enumerate(employees.items()):
                        for kdx in range(len(identity['identified_tracklets'])):
                            if key != identity['identified_tracklets'][kdx].tracklet_num:
                                l2 = np.sum( np.square(tracklet.person_embeds_mean -identity['identified_tracklets'][kdx].person_embeds_mean))+1e-12
#                                l2 = l2-40 if tracklet.cam_id!=identity['identified_tracklets'][kdx].cam_id else l2
                                self.embed_distance[jdx] = min( self.embed_distance[jdx] , l2 )                              
                    
                    sorted_distance   =  sorted(self.embed_distance)
                    if sorted_distance[0]<100:
                        dis_rank1_index = list(self.embed_distance).index(sorted_distance[0])
                        if list(employees.keys())[dis_rank1_index]!=tracklet.elan_id:
                            most_vote = list(employees.keys())[dis_rank1_index]
                            
#                            if most_vote in [ tracklet.elan_id for key, tracklet in tracklets.items()]:
#                                continue
#                            else:                                       
                            if 'guest' in tracklet.elan_id:
                                if guests == None:
                                    print('Please provide guest list in order to remove tracklet from guest list')
                                else:
                                    guests[tracklet.elan_id]['identified_tracklets'].remove(tracklet)
                                    result.append(('rmtr_gst', tracklet.elan_id, key))
                                    if len(guests[tracklet.elan_id]['identified_tracklets'])==0:
                                        del guests[tracklet.elan_id]
                                        result.append(('rm_gst', None, tracklet.elan_id))
                            tracklet.elan_id=most_vote
                            employees[most_vote]['identified_tracklets'].append(tracklet)
                            result.append(('adtr_emp', most_vote, key))
                            self.is_employee = True
                            
                    if self.is_employee ==False and len(guests)>1:
                        self.embed_distance   = np.ones(len(guests))*1000
                        for jdx, (elan_id,identity) in enumerate(guests.items()):
                            for kdx in range(len(identity['identified_tracklets'])):
                                if key != identity['identified_tracklets'][kdx].tracklet_num:
                                    self.embed_distance[jdx] = min( self.embed_distance[jdx] , 
                                                       np.sum( np.square(tracklet.person_embeds_mean -identity['identified_tracklets'][kdx].person_embeds_mean))+1e-12 )                
                                                       
                        sorted_distance   =  sorted(self.embed_distance)
                        if sorted_distance[0]<100:
                            dis_rank1_index = list(self.embed_distance).index(sorted_distance[0])
                            if list(guests.keys())[dis_rank1_index]!=tracklet.elan_id:
                                most_vote = list(guests.keys())[dis_rank1_index]
                                
                                if most_vote in [ tracklet.elan_id for key, tracklet in tracklets.items()]:
                                    continue
                                else:                            
                                    if 'guest' in tracklet.elan_id:
                                        if guests == None:
                                            print('Please provide guest list in order to remove tracklet from guest list')
                                        else:
                                            guests[tracklet.elan_id]['identified_tracklets'].remove(tracklet)
                                            result.append(('rmtr_gst', tracklet.elan_id, key))
                                            if len(guests[tracklet.elan_id]['identified_tracklets'])==0:
                                                del guests[tracklet.elan_id]
                                                result.append(('rm_gst', None, tracklet.elan_id))
                                    tracklet.elan_id=most_vote
                                    guests[most_vote]['identified_tracklets'].append(tracklet)
                                    result.append(('adtr_gst', most_vote, key))
                   
        return result
    
'''
Tracker: AlignReid 
Version: 1.0
Instruction:Identified identity based on tracklets with voting technique
Maintainer: C.H.Lu
'''  
class AlignReid_Identifier(Identifier):
    
    def __init__(self,):
        self.min_require_votes = 5
        self.guest_reid_thre = 0.5
        self.distance_max =10
        
    def identify(self, tracklets, employees, guests=None):        
        result =[]

        for key, tracklet in tracklets.items():
            if (tracklet.elan_id=='unidentified' or 'guest' in tracklet.elan_id):
                # Reidentify if tracklet is 'unidentified' or is 'guest'
                if len(tracklet.person_embeds) >= self.min_require_votes: 
                    # Check if tracklet belongs to any employee
                    self.is_employee = False
#                    self.embed_distance   = np.ones(len(employees))*1000
#                    for jdx, (elan_id,identity) in enumerate(employees.items()):
#                        for kdx in range(len(identity['identified_tracklets'])):
#                            if key != identity['identified_tracklets'][kdx].tracklet_num:
#                                self.embed_distance[jdx] = min( self.embed_distance[jdx] , 
#                                                   np.sum( np.square(tracklet.person_embeds_mean -identity['identified_tracklets'][kdx].person_embeds_mean))+1e-12 )                              
#                    sorted_distance   =  sorted(self.embed_distance)
#                    if sorted_distance[0]<81:
#                        dis_rank1_index = list(self.embed_distance).index(sorted_distance[0])
#                        if list(employees.keys())[dis_rank1_index]!=tracklet.elan_id:
#                            most_vote = list(employees.keys())[dis_rank1_index]
#                            
#                            if most_vote in [ tracklet.elan_id for key, tracklet in tracklets.items()]:
#                                continue
#                            else:                                       
#                                if 'guest' in tracklet.elan_id:
#                                    if guests == None:
#                                        print('Please provide guest list in order to remove tracklet from guest list')
#                                    else:
#                                        guests[tracklet.elan_id]['identified_tracklets'].remove(tracklet)
#                                        result.append(('rmtr_gst', tracklet.elan_id, key))
#                                        if len(guests[tracklet.elan_id]['identified_tracklets'])==0:
#                                            del guests[tracklet.elan_id]
#                                            result.append(('rm_gst', None, tracklet.elan_id))
#                                tracklet.elan_id=most_vote
#                                employees[most_vote]['identified_tracklets'].append(tracklet)
#                                result.append(('adtr_emp', most_vote, key))
#                                self.is_employee = True
                            
                    if self.is_employee ==False and len(guests)>1:
                        self.embed_distance   = np.ones(len(guests))*self.distance_max
                        for jdx, (elan_id,identity) in enumerate(guests.items()):
                            for kdx in range(len(identity['identified_tracklets'])):
                                if key != identity['identified_tracklets'][kdx].tracklet_num:
                                    self.embed_distance[jdx] = min( self.embed_distance[jdx] , 
                                                       np.sqrt(np.sum( np.square(tracklet.person_embeds_mean -identity['identified_tracklets'][kdx].person_embeds_mean))+1e-12 ))           
                                                       
                        sorted_distance   =  sorted(self.embed_distance)
                        if sorted_distance[0]<self.guest_reid_thre :
                            dis_rank1_index = list(self.embed_distance).index(sorted_distance[0])
                            if list(guests.keys())[dis_rank1_index]!=tracklet.elan_id:
                                most_vote = list(guests.keys())[dis_rank1_index]
                                
                                if most_vote in [ tracklet.elan_id for key, tracklet in tracklets.items()]:
                                    continue
                                else:                            
                                    if 'guest' in tracklet.elan_id:
                                        if guests == None:
                                            print('Please provide guest list in order to remove tracklet from guest list')
                                        else:
                                            guests[tracklet.elan_id]['identified_tracklets'].remove(tracklet)
                                            result.append(('rmtr_gst', tracklet.elan_id, key))
                                            if len(guests[tracklet.elan_id]['identified_tracklets'])==0:
                                                del guests[tracklet.elan_id]
                                                result.append(('rm_gst', None, tracklet.elan_id))
                                    tracklet.elan_id=most_vote
                                    guests[most_vote]['identified_tracklets'].append(tracklet)
                                    result.append(('adtr_gst', most_vote, key))
                   
        return result
    
