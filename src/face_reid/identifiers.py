# -*- coding: utf-8 -*-
"""
Abstract class for identifiers
Input_format : array of detections defined in src/detector/detectors.py
Output_format: array of detections defined in src/detector/detectors.py, filtered 
"""

import abc
class Identifier(abc.ABC):
 
    @abc.abstractmethod
    def identify(self):
        'Return Identifier result'
        return NotImplemented
    
import numpy as np


'''
Tracker: ArcFaceReid
Version: 1.1
Instruction:Identified identity based on tracklets with voting technique
Maintainer: C.H.Lu
'''
class ArcFaceReid(Identifier):
    def __init__(self,):
        self.min_require_votes = 7
    def identify(self, tracklets, employees, guests=None):
        result =[]
        for key, tracklet in tracklets.items():
            if tracklet.elan_id=='unidentified' or 'guest' in tracklet.elan_id:
                if len(tracklet.face_embeds) >= self.min_require_votes:
                    self.votes =[]
                    self.embed_distance   = np.ones ((len(tracklet.face_embeds),len(employees)))*10
                    self.embed_similarity = np.zeros((len(tracklet.face_embeds),len(employees)))
                    for idx in range(len(tracklet.face_embeds)):
                        for jdx, (elan_id, employee) in enumerate(employees.items()):
                            #self.embed_distance[idx,jdx]   = np.sqrt( np.sum( np.square(tracklet.face_embeds[idx] - employee['elan_seedfaceembed']))+1e-12 )
                            self.embed_similarity[idx,jdx] = np.dot( tracklet.face_embeds[idx], employee['elan_seedfaceembed'].T )                      
                        #sorted_distance   =  sorted(self.embed_distance[idx])
                        sorted_similarity =  sorted(self.embed_similarity[idx])
                        if sorted_similarity[-1]>0.5:# or sorted_distance[0]<1:
                            #dis_rank1_index = list(self.embed_distance[idx]).index(sorted_distance[0])
                            sim_rank1_index = list(self.embed_similarity[idx]).index(sorted_similarity[-1])
                            #if dis_rank1_index==sim_rank1_index:
                            self.votes.append( list(employees.keys())[sim_rank1_index])

                    if len(self.votes)>0:
                        #------votes =  1st - 2nd-------------------------
                        count_dict = dict((x,self.votes.count(x)) for x in set(self.votes))
                        sort_count = sorted(count_dict.items(), key=lambda item: item[1])
                        most_vote = sort_count[-1][0]
                        if len(sort_count)>1:
                            if sort_count[-2][0]!='unknown':
                                num_votes = sort_count[-1][1] - sort_count[-2][1]
                        else:
                            num_votes = sort_count[-1][1]
                        if num_votes >=self.min_require_votes:
                            
                        #------votes =  1st-------------------------------
                        #most_vote = max(set(self.votes), key=self.votes.count)
                        #if self.votes.count(most_vote) >=self.min_require_votes:

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

        return result
                

                