#Importing necessary libraries, mainly the OpenCV, and PyQt libraries
import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets

import src.video.videoCap as videocap
from src.gui.visualize import  ImageViewer

if __name__ == '__main__':
    # GUI app
    app = QtWidgets.QApplication(sys.argv)
    
    # Camera 1 
    # Camera Views 
    vid1_ImgView = ImageViewer()
    vid1_Thread  = QtCore.QThread()
    vid1_Cap  = videocap.VideoCap( 
                            video_path='./video/af_cam1.avi', 
                            cam_id=1, 
                            width=400, 
                            height=320, 
                            fps=15,
                            imgview=vid1_ImgView,
                            thread=vid1_Thread,
                            edge=None, 
                            parent=None)


    # Camera 2
    vid2_ImgView = ImageViewer()
    vid2_Thread  = QtCore.QThread()
    vid2_Cap  = videocap.VideoCap( 
                            video_path='./video/af_cam2.avi', 
                            cam_id=2, 
                            width=400, 
                            height=320, 
                            fps=30,
                            imgview=vid2_ImgView,
                            thread=vid2_Thread,
                            edge=None, 
                            parent=None)

    # Camera 3
    vid3_ImgView = ImageViewer()
    vid3_Thread  = QtCore.QThread()
    vid3_Cap  = videocap.VideoCap( 
                            video_path='./video/af_cam3.avi', 
                            cam_id=3, 
                            width=400, 
                            height=320, 
                            fps=15,
                            imgview=vid3_ImgView,
                            thread=vid3_Thread,
                            edge=None, 
                            parent=None)
 
    # Camera 4
    vid4_ImgView = ImageViewer()
    vid4_Thread  = QtCore.QThread()
    vid4_Cap  = videocap.VideoCap( 
                            video_path='./video/af_cam4.avi', 
                            cam_id=4, 
                            width=400, 
                            height=320, 
                            fps=30,
                            imgview=vid4_ImgView,
                            thread=vid4_Thread,
                            edge=None, 
                            parent=None)

    # Camera 5
    vid5_ImgView = ImageViewer()
    vid5_Thread  = QtCore.QThread()
    vid5_Cap  = videocap.VideoCap( 
                            video_path='./video/af_cam5.avi', 
                            cam_id=5, 
                            width=400, 
                            height=320, 
                            fps=15,
                            imgview=vid5_ImgView,
                            thread=vid5_Thread,
                            edge=None, 
                            parent=None)

    # Camera 6
    vid6_ImgView = ImageViewer()
    vid6_Thread  = QtCore.QThread()
    vid6_Cap  = videocap.VideoCap( 
                            video_path='./video/af_cam6.avi', 
                            cam_id=6, 
                            width=400, 
                            height=320, 
                            fps=30,
                            imgview=vid6_ImgView,
                            thread=vid6_Thread,
                            edge=None, 
                            parent=None)

    # Camera 7
    vid7_ImgView = ImageViewer()
    vid7_Thread  = QtCore.QThread()
    vid7_Cap  = videocap.VideoCap( 
                            video_path='./video/af_cam7.avi', 
                            cam_id=7, 
                            width=400, 
                            height=320, 
                            fps=30,                          
                            imgview=vid7_ImgView,
                            thread=vid7_Thread,
                            edge=None, 
                            parent=None)

#    # Camera 8
#    vid8_ImgView = ImageViewer()
#    vid8_Thread  = QtCore.QThread()
#    vid8_Cap  = videocap.VideoCap( 
#                            video_path='./video/af_cam8.avi', 
#                            cam_id=8, 
#                            width=400, 
#                            height=320, 
#                            imgview=vid8_ImgView,
#                            thread=vid8_Thread,
#                            edge=None, 
#                            parent=None) 
#
#    # Camera 9    
#    vid9_ImgView = ImageViewer()
#    vid9_Thread  = QtCore.QThread()
#    vid9_Cap  = videocap.VideoCap( 
#                            video_path='./video/af_cam9.avi', 
#                            cam_id=9, 
#                            width=400, 
#                            height=320, 
#                            imgview=vid9_ImgView,
#                            thread=vid9_Thread,
#                            edge=None, 
#                            parent=None)  


    # Start button
    push_button1= QtWidgets.QPushButton('Start')
    # Link the start button to camera function
    push_button1.clicked.connect(vid1_Cap.start)
    push_button1.clicked.connect(vid2_Cap.start)
    push_button1.clicked.connect(vid3_Cap.start)
    push_button1.clicked.connect(vid4_Cap.start)
    push_button1.clicked.connect(vid5_Cap.start)
    push_button1.clicked.connect(vid6_Cap.start)
    push_button1.clicked.connect(vid7_Cap.start)
#    push_button1.clicked.connect(vid8_Cap.start)
#    push_button1.clicked.connect(vid9_Cap.start)
    
    # Main Layout of the GUI
    form_layout = QtWidgets.QFormLayout()
    
    # Row1
    hbox_row1 = QtWidgets.QHBoxLayout()
    hbox_row1.addWidget(vid1_ImgView)
    hbox_row1.addWidget(vid4_ImgView)
    hbox_row1.addWidget(vid7_ImgView)
    
    # Row2   
    hbox_row2 = QtWidgets.QHBoxLayout()
    hbox_row2.addWidget(vid2_ImgView)
    hbox_row2.addWidget(vid5_ImgView)
#    hbox_row2.addWidget(vid8_ImgView)
    
    # Row3
    hbox_row3 = QtWidgets.QHBoxLayout()
    hbox_row3.addWidget(vid3_ImgView)
    hbox_row3.addWidget(vid6_ImgView)
#    hbox_row3.addWidget(vid9_ImgView)
    
    # Main Layout add Row1/2/3
    form_layout.addRow(hbox_row1)
    form_layout.addRow(hbox_row2)
    form_layout.addRow(hbox_row3)    
    form_layout.addRow(push_button1)
    

    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(form_layout)
 
    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    exc = app.exec_()
