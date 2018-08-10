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
    
        
import cv2
import numpy as np

class Lumin_Filter(Filter):    
    def __init__(self,):
        pass
    def filter_out(self, detections, visual_only=False):        
        output=[]        
        for det in detections: 
            bbox = det['person_img']
            luminance = 0.2126*bbox[:,:,0]+0.7152*bbox[:,:,1]+0.0722*bbox[:,:,2]
            luminance = np.mean(luminance)
            if luminance>60:
                output.append(det)
            else:
                if visual_only==True:
                    det['person_img'] = cv2.putText(det['person_img'].copy(), str(luminance), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1  , (255,0,0), 2)
                    det['person_img'] = cv2.line(det['person_img'].copy(),(0,0),(det['person_img'].shape[1],det['person_img'].shape[0]),(255,0,0),5,1)
                    det['person_img'] = cv2.line(det['person_img'].copy(),(det['person_img'].shape[1],0),(0,det['person_img'].shape[0]),(255,0,0),5,1)
                    output.append(det)
        return output


# TEST CODES
if __name__ == '__main__':
    import os
    os.chdir('../../')   
    import src.detector.detectors as detectors
    import matplotlib.pyplot as plt 

    Y3 = detectors.Yolov3()
    os.chdir('./src/person_reid')  
    test_img = cv2.imread('test.png')
    vis_img = test_img.copy()
    detections = Y3.predict(test_img)
    for track in detections:
        x1, y1, x2, y2 = track['person_bbox']
        color = np.random.randint(low=0,high=255,size=3)    
        color = (int(color[0]),int(color[1]),int(color[2]))
        cv2.rectangle(vis_img,(x1, y1), (x2, y2),color,5)
    plt.imshow(vis_img[:,:,::-1])
    plt.show()

   
    F = Lumin_Filter()
    
    import time
    s=  time.time()
    detections = F.filter_out(detections, visual_only=True)
    e=  time.time()
    print(e-s)
    for track in detections:
        color = np.random.randint(low=0,high=255,size=3)    
        color = (int(color[0]),int(color[1]),int(color[2]))
        if track['person_img'] is not None:   
            plt.imshow(track['person_img'][:,:,::-1])
            plt.show() 
        



