#Importing necessary libraries, mainly the OpenCV, and PyQt libraries
import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from src.gui.visualize import  ImageViewer, ScrollImageViewer, TrackletBorad , IdentityBoard

import src.detector.detectors as detectors
import src.tracker.trackers as trackers
import src.person_reid.filters as filters
import src.person_reid.embedders as embedders
import src.person_reid.identifiers as identifiers
import src.face_reid.filters as face_filters
import src.face_reid.embedders as face_embedders
import src.face_reid.identifiers as face_identifiers
import src.server.server as server
import src.server.analyzer as analyzer
import src.edge.edge as edge
import src.video.videoCap as videocap

#-------------Modules (With    Face)-----------------------------------------------------
Detector                 = detectors.Yolov3_MTCNN_Batch()
PersonReID_Filter        = None#filters.Lumin_Filter()
PersonReID_Embedder      = embedders.Triplet_Embedder()
FaceReID_Filter          = None#face_filters.Frank_Filter()
FaceReID_Embedder        = face_embedders.ArcFace_Embedder()

Tracker                  = trackers.Triplet_w_IOUTracker()
PersonReID_Identifier    = identifiers.TripletReid()
FaceReID_Identifier      = face_identifiers.ArcFaceReid()

#------------Pseudo Server---------------------------------------------------------------

Pseudo_Server            = server.Server( tracker=Tracker)      
Pseudo_Server.initial_employees(imgpath='./image/FACE_ID/', detector=Detector, embedder=FaceReID_Embedder, load_if_exist=True)
Analyzer                 = analyzer.Analyzer( src_server=Pseudo_Server, face_identifier=FaceReID_Identifier, person_identifier=PersonReID_Identifier)  



    
if __name__ == '__main__':
 
    app = QtWidgets.QApplication(sys.argv)
    
    # Camera Views 
    vid1_ImgView = ImageViewer()
    vid1_Thread  = QtCore.QThread()
    vid1_Edge    = edge.Edge(
                             server=Pseudo_Server, 
                             detector=Detector, 
                             person_filter=PersonReID_Filter, 
                             person_aligner=None, 
                             person_embedder=PersonReID_Embedder, 
                             face_filter=FaceReID_Filter, 
                             face_aligner=None, 
                             face_embedder=FaceReID_Embedder, 
                             parent=None)

    vid1_Cap  = videocap.VideoCap( 
                            video_path='./video/af_cam10f_1.mp4', 
                            cam_id=1, 
                            width=1200, 
                            height=960, 
                            imgview=vid1_ImgView,
                            thread=vid1_Thread,
                            edge=vid1_Edge, 
                            parent = None)

    
    # Tracking Board
    trackletBrd_ImgView = ImageViewer()
    trackletBrd_Thread  = QtCore.QThread()
    trackletBrd         = TrackletBorad(Pseudo_Server, trackletBrd_ImgView, trackletBrd_Thread)
 
    
    # Employee Board
    employeebrd_ImgView = ScrollImageViewer(w=600, h=1000)
    employeebrd_Thread  = QtCore.QThread()
    employeebrd         = IdentityBoard(Pseudo_Server.S_employees, employeebrd_ImgView, employeebrd_Thread)


    # Guest Board
    guestbrd_ImgView    = ScrollImageViewer(w=600, h=1000)
    guestbrd_Thread     = QtCore.QThread()
    guestbrd            = IdentityBoard(Pseudo_Server.S_guests, guestbrd_ImgView, guestbrd_Thread)

    
    
    
    # Linking Start Button
    push_button1 = QtWidgets.QPushButton('Start')
    push_button1.clicked.connect(vid1_Cap.start)
    push_button1.clicked.connect(guestbrd.start)
    push_button1.clicked.connect(employeebrd.start)
    push_button1.clicked.connect(trackletBrd.start)
    push_button1.clicked.connect(Analyzer.start)
    
    
    # GUI Layout
    form_layout = QtWidgets.QFormLayout()
    hbox_row1 = QtWidgets.QHBoxLayout()
    hbox_row1.addWidget(vid1_ImgView)
       
    form_layout.addRow(hbox_row1)
    form_layout.addRow(push_button1)
        
    main_layout = QtWidgets.QHBoxLayout()
    main_layout.addLayout(form_layout)    
    main_layout.addWidget(trackletBrd_ImgView)
    main_layout.addWidget(guestbrd_ImgView)
    main_layout.addWidget(employeebrd_ImgView)
    
    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(main_layout)
 
    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    app.exec_()
    
    
    Analyzer.runnable = False
    vid1_Cap.stop()