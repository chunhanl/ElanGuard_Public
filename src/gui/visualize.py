# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

'''
Class: ImageViewer
Usage: Display Images on GUI
'''
class ImageViewer(QtWidgets.QWidget):
    def __init__(self, w=400, h=320, parent = None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self.init_w = w
        self.init_h = h
        
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0,0, self.image)
        self.image = QtGui.QImage()
 
    def initUI(self):
        self.setWindowTitle('Test')
        
    @QtCore.pyqtSlot()
    def setDefaultImage(self,default_img=None):
        if np.any(default_img)!=None:
            def_img = cv2.resize(default_img,(self.init_w,self.init_h))
        else:
            def_img = cv2.resize(np.zeros((100,100,3),dtype=np.uint8),(self.init_w,self.init_h))
        height, width, _ = def_img.shape
        qt_image = QtGui.QImage(def_img.data,
                                width,
                                height,
                                def_img.strides[0],
                                QtGui.QImage.Format_RGB888)
        self.image = qt_image
        if self.image.size() != self.size():
            self.setFixedSize(qt_image.size())
        self.update()
    
    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()
 
'''
Class: ScrollImageViewer
Usage: Display Images on GUI
'''
class ScrollImageViewer(QtWidgets.QScrollArea):
    
    def __init__(self, w=600, h=1000, parent = None):
        super(ScrollImageViewer, self).__init__(parent)
        self.imgviewer = ImageViewer()
        self.setWidget(self.imgviewer)
        self.setWidgetResizable(True)
        self.setFixedHeight(h)
        self.setFixedWidth(w)

    @QtCore.pyqtSlot()
    def setDefaultImage(self,default_img=None):
        self.imgviewer.setDefaultImage(default_img)
    
    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        self.imgviewer.setImage(image)
'''
Class: TrackletBorad
Usage: Display Person Tracklets on GUI
'''
class TrackletBorad(QtCore.QObject):    
    TrackletBoradSignal = QtCore.pyqtSignal(QtGui.QImage)
    
    def __init__(self, src_server, imgview, thread, parent = None):
        super(TrackletBorad, self).__init__(parent)
        self.server = src_server
        self.imgview = imgview
        self.thread = thread
        self.update_period = 1 # second
        self.height = 1000
        self.width = 500
        self.num_id = 10
        self.num_bbox = 10
        self.height_per_id  = self.height // self.num_id
        self.width_per_bbox = self.width // self.num_bbox 

        self.moveToThread(self.thread)
        self.TrackletBoradSignal.connect(self.imgview.setImage)      
        self.thread.start()    
        
    @QtCore.pyqtSlot()
    def start(self):
        run = True
        while run:
            display_img = np.zeros((self.height,self.width,3),dtype=np.uint8)
            # show the latest id in S_tracklets
            showing_ids = list(self.server.S_tracklets.keys())[-self.num_id::]
            # show the id in active detections
            for cam_id, dets in self.server.S_active_detections.items():
                for det in dets[-1]:
                    if det['s_tracklet_num'] not in showing_ids:
                        showing_ids.append(det['s_tracklet_num'])
            showing_ids = showing_ids[-self.num_id::]
            
            for idx, display_id in enumerate(showing_ids):
                tracklet = self.server.S_tracklets.get_tracklet(display_id)
                pos_y = idx%self.num_id*self.height_per_id  
                tr_id =  str(display_id)
                #tr_cam = 'Cam'+str(track['cam'])
                color = (int(tracklet.color[0]),int(tracklet.color[1]),int(tracklet.color[2]))
                # show the 10 latest bboxs 
                start_jdx = -1 * min(len(tracklet.person_imgs),self.num_bbox)
                for jdx, det in enumerate(tracklet.person_imgs[start_jdx::]):
                    pos_x = jdx%self.num_bbox*self.width_per_bbox
                    if jdx==0:
                        bbox = cv2.resize(tracklet.person_imgs[0],(self.width_per_bbox,self.height_per_id))
                        cv2.putText(bbox, tr_id, (2,self.height_per_id//2), cv2.FONT_HERSHEY_SIMPLEX, 1  , color, 2)
                        cv2.rectangle(bbox,(0, 0),(self.width_per_bbox,self.height_per_id),color,3)
                    else:
                        bbox = cv2.resize(det,(self.width_per_bbox,self.height_per_id))

                    display_img[pos_y:pos_y+self.height_per_id,pos_x:pos_x+self.width_per_bbox,:] = bbox
  
            
#            cv2.imwrite('tr_'+str(time.time())+'.jpg',display_img[:,:,::-1])
            height, width, _ = display_img.shape
            qt_image = QtGui.QImage(display_img.data,
                                    width,
                                    height,
                                    display_img.strides[0],
                                    QtGui.QImage.Format_RGB888)
 
            self.TrackletBoradSignal.emit(qt_image) 
            time.sleep(self.update_period)



'''
Class: IdentityBoard
Usage: Display Face Identification Result
'''
class IdentityBoard(QtCore.QObject):    
    ImageSignal = QtCore.pyqtSignal(QtGui.QImage)
    
    def __init__(self, identities_dict, imgview, thread, parent = None):
        super(IdentityBoard, self).__init__(parent)
        self.update_period = 0.3 # second
        self.num_col = 1
        self.width = 1000
        self.width_per_id_col  = self.width // self.num_col
        self.height_per_id_slot  = 125
        
        self.identities_dict = identities_dict
        self.imgview = imgview
        self.thread = thread
        
        self.moveToThread(self.thread)
        self.ImageSignal.connect(self.imgview.setImage)      
        self.thread.start()    
        
    @QtCore.pyqtSlot()
    def start(self):
        run = True
        while run:
            num_of_id = len(self.identities_dict.keys())
            self.height = self.height_per_id_slot*num_of_id
            idBrd_img = np.ones((self.height,self.width,3),dtype=np.uint8)*255
            
            num_shown = 0
            for col in range(self.num_col):
                for i in range(num_of_id):
                    pos_x = col*self.width_per_id_col
                    pos_y = self.height_per_id_slot*num_shown
                    slot_img = np.ones((self.height_per_id_slot,self.width_per_id_col,3),dtype=np.uint8)*128
                    num_of_id = len(self.identities_dict.keys())
                    if i+col*8<num_of_id:   
                        elan_id = list(self.identities_dict.keys())[i+col*8]
                        show_identity = self.identities_dict[elan_id]
                        if len(show_identity['identified_tracklets'])>0: 
                            num_shown+=1
                            if show_identity['elan_photo'] is not None:
                                face_img = show_identity['elan_photo']
                                face_img = cv2.resize(face_img,(100,self.height_per_id_slot))
                            else:
                                face_img = np.zeros((self.height_per_id_slot, 100, 3))
                                
                            color = (220,20,60) if 'guest' in elan_id else (0,255,0)
                            face_img = cv2.putText(face_img, elan_id, (0,self.height_per_id_slot-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            slot_img[0:self.height_per_id_slot,0:100] = face_img
                                                       
                            for j,cap_face_img in enumerate(show_identity['identified_tracklets'][-1].face_imgs[-8::]):
                                face_pos_x = 100+60*(j%4)
                                face_pos_y = j//4*self.height_per_id_slot//2
                                cap_face_img = cv2.resize(cap_face_img,(60,self.height_per_id_slot//2))
                                slot_img[face_pos_y:face_pos_y+self.height_per_id_slot//2,face_pos_x:face_pos_x+60] = cap_face_img
                        
                            start_jdx = -1 * min(len(show_identity['identified_tracklets']),4)
                            for j,tracklet in enumerate(show_identity['identified_tracklets'][start_jdx::]):
                                per_pos_x = 340+60*j
                                cap_pers_img = cv2.resize(tracklet.person_imgs[-1], (60,self.height_per_id_slot))
                                color = tracklet.color                            
                                cv2.rectangle(cap_pers_img,(0, 0),(60,self.height_per_id_slot),color,3)
                                slot_img[0:self.height_per_id_slot,per_pos_x:per_pos_x+60] = cap_pers_img

                    idBrd_img[pos_y:pos_y+self.height_per_id_slot,pos_x:pos_x+self.width_per_id_col] = slot_img
                
#            cv2.imwrite('id_'+str(time.time())+'.jpg',idBrd_img[:,:,::-1])
            height, width, _ = idBrd_img.shape
            qt_image = QtGui.QImage(idBrd_img.data,
                                    width,
                                    height,
                                    idBrd_img.strides[0],
                                    QtGui.QImage.Format_RGB888)
 
            self.ImageSignal.emit(qt_image) 
            time.sleep(self.update_period)
            
                     

'''
Class: IdBoard
Usage: Display Face Identification Result
'''
'''
class IdBoard(QtCore.QObject):    
    IdBoardSignal = QtCore.pyqtSignal(QtGui.QImage)
           
    def __init__(self, src_server, parent = None):
        super(IdBoard, self).__init__(parent)
        self.update_period = 0.3 # second
        self.height = 1000
        self.width = 1800
        self.num_id_slot = 8
        self.num_col = 3
        self.height_per_id_slot  = self.height // self.num_id_slot
        self.width_per_id_col  = self.width // self.num_col
        self.server = src_server
        self.employees_dict = self.server.S_employees
        
    @QtCore.pyqtSlot()
    def startIdboard(self):
        run = True
        while run:
            idBrd_img = np.ones((self.height,self.width,3),dtype=np.uint8)*255
            
            for col in range(self.num_col):
                for i in range(self.num_id_slot):
                    pos_x = col*self.width_per_id_col
                    pos_y = self.height_per_id_slot*i
                    slot_img = np.ones((self.height_per_id_slot,self.width_per_id_col,3),dtype=np.uint8)*128
                    if i+col*8<len(self.employees_dict.keys()):
                        elan_id = list(self.employees_dict.keys())[i+col*8]
                        
                        face_img = self.employees_dict[elan_id]['elan_photo']
                        face_img = cv2.resize(face_img,(100,self.height_per_id_slot))
                        face_img = cv2.putText(face_img, elan_id, (0,self.height_per_id_slot-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        slot_img[0:self.height_per_id_slot,0:100] = face_img
                        
                        if len(self.employees_dict[elan_id]['identified_tracklets'])>0:                                
                            for j,cap_face_img in enumerate(self.employees_dict[elan_id]['identified_tracklets'][-1].face_imgs[-8::]):
                                face_pos_x = 100+60*(j%4)
                                face_pos_y = j//4*self.height_per_id_slot//2
                                cap_face_img = cv2.resize(cap_face_img,(60,self.height_per_id_slot//2))
                                slot_img[face_pos_y:face_pos_y+self.height_per_id_slot//2,face_pos_x:face_pos_x+60] = cap_face_img
                        
                        if len(self.employees_dict[elan_id]['identified_tracklets'])>0:
                            start_jdx = -1 * min(len(self.employees_dict[elan_id]['identified_tracklets']),4)
                            for j in range(len(self.employees_dict[elan_id]['identified_tracklets'])):
                                per_pos_x = 340+60*j
                                cap_pers_img = cv2.resize(self.employees_dict[elan_id]['identified_tracklets'][start_jdx+j].person_imgs[-1],(60,self.height_per_id_slot))
                                color = self.employees_dict[elan_id]['identified_tracklets'][start_jdx+j].color                            
                                cv2.rectangle(cap_pers_img,(0, 0),(60,self.height_per_id_slot),color,3)
                                slot_img[0:self.height_per_id_slot,per_pos_x:per_pos_x+60] = cap_pers_img
#                            for j,cap_pers_img in enumerate(self.employees_dict[i+col*8]['identified_tracklets'][-1]['person_imgs'][-4::]):
#                                face_pos_x = 340+60*j
#                                cap_pers_img = cv2.resize(cap_pers_img,(60,self.height_per_id_slot))
#                                slot_img[0:self.height_per_id_slot,face_pos_x:face_pos_x+60] = cap_pers_img

                    idBrd_img[pos_y:pos_y+self.height_per_id_slot,pos_x:pos_x+self.width_per_id_col] = slot_img
                
#            cv2.imwrite('id_'+str(time.time())+'.jpg',idBrd_img[:,:,::-1])
            height, width, _ = idBrd_img.shape
            qt_image = QtGui.QImage(idBrd_img.data,
                                    width,
                                    height,
                                    idBrd_img.strides[0],
                                    QtGui.QImage.Format_RGB888)
 
            self.IdBoardSignal.emit(qt_image) 
            time.sleep(self.update_period)

'''
