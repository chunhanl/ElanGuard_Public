# -*- coding: utf-8 -*-
"""
Abstract class for preprocessor
"""
import abc
class Preprocessor(abc.ABC):
       
    @abc.abstractmethod
    def process(self):
        'Return processed frames'
        return NotImplemented



#import cv2
#class Auto_White_Balance(Preprocessor):
#    def __init__(self,):
#        self.inst = xphoto.createGrayworldWB()
#        self.inst.setSaturationThreshold(0.95)
#
#    def process(self, image):
#        return self.inst.balanceWhite(image)

    
## TEST CODES
#if __name__ == '__main__':
#    import matplotlib.pyplot as plt
#    import time 
#    
#    P = Auto_White_Balance()
#    testimg=cv2.imread("test.png")
#    testimgrgb=cv2.cvtColor(testimg, cv2.COLOR_BGR2RGB)
#    
#    plt.imshow(testimgrgb)
#    plt.show()
#    s=time.time()
#    aftestimg=P.process(testimg)
#    e=time.time() 
#    print('Time:', e-s)
#    
#    aftestimg=cv2.cvtColor(aftestimg, cv2.COLOR_BGR2RGB)
#    plt.imshow(aftestimg)
#    plt.show()