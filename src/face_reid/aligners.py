# -*- coding: utf-8 -*-
"""
Abstract class for aligners
Input_format : array of detections defined in src/detector/detectors.py
Output_format: array of detections defined in src/detector/detectors.py, 
               WITH 'face_img_aligned' filled if 'face_img' and 'face_keypoint' is not None
"""
import abc
class Aligner(abc.ABC):
    @abc.abstractmethod
    def align(self):
        'Return filtered detections'
        return NotImplemented
    

class DoNothing_Aligner(Aligner):
    def __init__(self,):
        print('Do Nothing Aligner Initialized')

    def align(self, detections):
        return detections