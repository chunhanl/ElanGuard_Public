# -*- coding: utf-8 -*-
"""
Abstract class for filters
Input_format : array of detections defined in src/detector/detectors.py
Output_format: array of detections defined in src/detector/detectors.py, filtered 
"""
import abc
class Filter(abc.ABC):
    @abc.abstractmethod
    def filter_out(self):
        'Return filtered detections'
        return NotImplemented
    

class AllPass_Filter(Filter):
    def __init__(self,):
        print('All Pass Filter Initialized')

    def filter_out(self, detections):
        return detections
    
    
from shapely.geometry import Polygon
from shapely.geometry import Point    
import cv2
import numpy as np

class Frank_Filter(Filter):    
    def __init__(self,):
        self.min_hwsize=32
#        self.hw_ratio=0.7
        self.min_brenner=50
        self.face_cascade = cv2.CascadeClassifier('./src/face_reid/haarcascade_frontalface_default.xml')
        
    def filter_out(self, detections, visual_only=False):   
        
        for det in detections:
            if det['face_bbox'] is not None:
                Pass=False
                facew=det['face_bbox'][3]-det['face_bbox'][1]
                faceh=det['face_bbox'][2]-det['face_bbox'][0]
                
                if facew>self.min_hwsize and faceh>self.min_hwsize:
                    
#                    if (facew/faceh)>self.hw_ratio and (faceh/facew)>self.hw_ratio:
                        
                    kernel = np.array([-1,0,1])
                    grayimg=cv2.cvtColor(det['face_img'], cv2.COLOR_RGB2GRAY)
                    dst=cv2.filter2D(np.float32(grayimg),-1,kernel)
                    dst2=dst*dst
                    if np.mean(dst2)>self.min_brenner:
                        
                        poly = Polygon((det['face_keypoint'][0], det['face_keypoint'][1], det['face_keypoint'][4], det['face_keypoint'][3]))
                        nosep=Point(det['face_keypoint'][2])
                        if nosep.within(poly):
                            Pass=True  
#                            grayimg=cv2.cvtColor(det['face_img'], cv2.COLOR_RGB2GRAY)
#                            self.isvjface=np.array(self.face_cascade.detectMultiScale(grayimg))
#                            if self.isvjface.size!=0:
#                                Pass=True  
#                            else:
#                                Situation='VJface'                                         
                        else:
                            Situation='Angle'
                    else:
                        Situation='Blur'
#                    else:
#                        Situation='Ratio'
                else:
                    Situation='Size'


                if Pass==False:
                    if visual_only==True:
                        det['face_img'] = cv2.line(det['face_img'],(0,0),(det['face_img'].shape[1],det['face_img'].shape[0]),(255,0,0),5,1)
                        det['face_img'] = cv2.line(det['face_img'],(det['face_img'].shape[1],0),(0,det['face_img'].shape[0]),(255,0,0),5,1)
                        det['face_img'] = cv2.putText(det['face_img'],Situation, (0,det['face_img'].shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    else:
                        det['face_img']  = None
                        det['face_bbox'] = None
                        det['face_confidence'] = None
                        det['face_keypoint']   = None
                        
        return detections




# TEST CODES
if __name__ == '__main__':
    import os
    os.chdir('../../')   
    import src.detector.detectors as detectors
    import matplotlib.pyplot as plt 
    # Detector and Embedder
    MTCNN = detectors.MTCNN()     
    F = Frank_Filter()
    os.chdir('./src/face_reid')  
    
    
    test_img = cv2.imread('test.jpg')
    vis_img = test_img.copy()
    detections = MTCNN.predict(test_img)
    for track in detections:
        if track['face_bbox'] is not None:
            x1, y1, x2, y2 = track['face_bbox']
            color = np.random.randint(low=0,high=255,size=3)    
            color = (int(color[0]),int(color[1]),int(color[2]))
            cv2.rectangle(vis_img,(x1, y1), (x2, y2),color,5)
    plt.imshow(vis_img[:,:,::-1])
    plt.show()  
    
    import time
    s=  time.time()
    detections = F.filter_out(detections, visual_only=True)
    e=  time.time()
    print(e-s)
    for track in detections:
        color = np.random.randint(low=0,high=255,size=3)    
        color = (int(color[0]),int(color[1]),int(color[2]))
        if track['face_img'] is not None:         
           vis_img = track['face_img'].copy()
           for pt in track['face_keypoint']:
               vis_img = cv2.circle(vis_img, (pt[0]-track['face_bbox'][0], pt[1]-track['face_bbox'][1]), 1, color,3 ,1)
           plt.imshow(vis_img[:,:,::-1])
           plt.show() 
        



