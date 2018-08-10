# -*- coding: utf-8 -*-
import time 
import cv2
from PyQt5 import QtCore
from PyQt5 import QtGui
class VideoCap(QtCore.QObject):    
    
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
    
    def __init__(self, video_path, cam_id=None, width=400, height=320, fps=15,
                     imgview=None,
                     thread=None,
                     edge=None, 
                     parent = None ):
        
        super(VideoCap, self).__init__(parent)
        self.cam_id = cam_id
        self.video_path = video_path
        self.width = width
        self.height = height
        self.is_running = False
        self.max_fps = fps
        self.min_per = 1.0/self.max_fps
        
        self.imgview = imgview
        self.thread = thread
        self.EDGE = edge                

        self.moveToThread(self.thread)
        self.VideoSignal.connect(imgview.setImage)
        self.thread.start()    
    
    @QtCore.pyqtSlot()
    def start(self):
        
        self.camera_port = 0
        self.camera = cv2.VideoCapture(self.camera_port)
        self.camera.open(self.video_path)
        self.camera.set(cv2.CAP_PROP_POS_FRAMES,0)

        self.frame_count = 0        
        self.is_running = True
        
        self.centers = {}
        self.record = {}
        while self.is_running:

            time_start = time.time()            
            ret, image = self.camera.read()
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      
            
            if self.EDGE is not None:
                # Feature Extraction
                detections = self.EDGE.pack_features(image, cam_id=self.cam_id,frame_no=self.frame_count)
    
                # Visualization
                for det in detections:
                    x1, y1, x2, y2 = det['person_bbox']
                    color = (int(det['s_tracklet_color'][0]),int(det['s_tracklet_color'][1]),int(det['s_tracklet_color'][2]))
                    cv2.rectangle(image,(x1, y1),(x2, y2),color,2)
                    cv2.putText(image,str(det['s_tracklet_num']), (x1+0,y1+25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    if det['face_bbox']!=None:
                        fx1, fy1, fx2, fy2 = det['face_bbox']
                        cv2.rectangle(image, (x1+fx1, y1+fy1), (x1+fx2, y1+fy2), color, 2)
                        for pt in det['face_keypoint']:
                            cv2.circle(image, (x1+pt[0], y1+pt[1]), 1, color,2 ,2)
                    if det['s_elan_id']=='unidentified':
                        color = (255,255,255)
                    elif 'guest' in det['s_elan_id']:
                        color = (220,20,60)
                    else:
                        color = (0,255,0)
                    cv2.putText(image,det['s_elan_id'].split('.')[0], (x1+(x2-x1)//2,y1+25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    

                
            image = cv2.resize(image,(self.width, self.height))
            height, width, _ = image.shape
            qt_image = QtGui.QImage(image.data,
                                    width,
                                    height,                    
                                    image.strides[0],
                                    QtGui.QImage.Format_RGB888)

            self.VideoSignal.emit(qt_image)
            time_end = time.time() 
            # Maintain fps
            if (time_end-time_start)<self.min_per:
                time.sleep(self.min_per-(time_end-time_start))

            self.frame_count+=1
            
    def stop(self):
        self.is_running = False
        self.camera.release()
        
        
        
        
        
        