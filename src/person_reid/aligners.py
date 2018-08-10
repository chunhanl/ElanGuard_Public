# -*- coding: utf-8 -*-
"""
Abstract class for aligners
Still have no idea how to use this part
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