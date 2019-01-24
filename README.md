# ElanGuard Readme

## System Design
* Flow Chart
![](https://i.imgur.com/CdoaQFa.png)    

* Edge Overview
![](https://i.imgur.com/grgDfMI.png)

* Edge Dataflow
![](https://i.imgur.com/Q0sZyDq.png)
  
* Server Overview
![](https://i.imgur.com/NXoUymS.png)    

* Server Dataflow
![](https://i.imgur.com/8hOTOwc.png)    


## Install

### 1. Install Anaconda  
Anaconda is highly recommanded
Otherwise may have to install the dependencies in **environment.yml** manually   

For Anaconda, simply create the environment with **environment.yml** :
`conda env create -f environment.yml`   
(For details, see [Creating an environment from an environment.yml file.](https://conda.io/docs/user-guide/tasks/manage-environments.html#create-env-file-manually))

After installed, enter the environment with
`source activate ElanGuard`   


### 2. Install some of the modules if needed
* Yolov2         
    `cd ./src/detector/darkflow/`  
    `pip install .`  
    https://github.com/thtrieu/darkflow  

* Yolov3           
    `pip3 install yolo34py-gpu`  
    https://github.com/madhawav/YOLO3-4-Py  

* Convolutional Pose Machine         
    `cd ./src/detector/tf-pose-estimation/`  
    `pip3 install -r requirements.txt`  
    https://github.com/ildoonet/tf-pose-estimation  

* ArcFace         
    `pip install mxnet-cu80 --pre`  
    `pip install easydict`  
    

### 3. (Optional) Install [Spyder IDE]  
`source activate ElanGuard`  
`conda install spyder`  
`spyder`  

(https://en.wikipedia.org/wiki/Spyder_(software)) for developing  
Spyder provides an easy way to visualize the **data structure**   

![](https://imgur.com/a/qtrikgg)


## Download
models, videos can be downloaded in [Google Drive](https://drive.google.com/drive/u/1/folders/0AOm-AQbL-WhpUk9PVA)  
( Please contact us if permission for access is needed )


## Demo Codes (GPU REQUIRED)
* **gui_multicam.py**  
Example of building up a muti-camera view GUI  
![image](https://github.com/chunhanl/ElanGuard_Public/blob/master/image/DemoGif/gui_multicam.gif)  

* **gui_singleview_personReID.py**  
Example of tracking and identifying guest in single camera view
![image](https://github.com/chunhanl/ElanGuard_Public/blob/master/image/DemoGif/gui_singleview_personReID.gif)  


* **gui_singleview_personReID_faceReID.py**  
Example of tracking and identifying employee in single camera view
![image](https://github.com/chunhanl/ElanGuard_Public/blob/master/image/DemoGif/gui_singleview_personReID_faceReID.gif)  



## Computation Consumption
| Module     | Algorithm/Model  | Code Version | Maintainer | Last update | Data Memory | GPU Memory | Computation Time(ms/frame)                 |
| :--------: | :--------------: | :----------: | :--------: | :---------: | :---------: | :--------: | :------------------------:                 | 
| Detector   | MTCNN            | 1.0          |            | 2018.06.04  | :---------: | :--------: |                                            |
| Detector   | MTCNN_Batch      | 1.0          |            | 2018.06.04  | :---------: | :--------: | 50                                         |
| Detector   | MTCNN_Batch_Fixed| 1.0          |            | 2018.06.05  | :---------: | :--------: | 10                                         |
| Detector   | Yolov2           | 1.1          |            | 2018.05.21  | :---------: | :--------: | 4                                          | 
| Detector   | Yolov2+MTCNN     | 1.1          |            | 2018.05.21  | :---------: | :--------: | 30                                         |
| Detector   | Yolov3           | 1.0          |            | 2018.06.04  | :---------: | :--------: | 90                                         | 
| Detector   | Yolov3+MTCNN     | 1.0          |            | 2018.06.04  | :---------: | :--------: |                                            |
| Detector   | MaskRCNN         | 1.1          |            | 2018.05.21  | :---------: | :--------: | 30                                         | 
| Detector   | ConvPoseMachine  | 1.0          |            | 2018.05.09  | :---------: | :--------: |                                            | 
| | | | | | | | | 
| Tracker    | TripletTracker   | 1.1          |            | 2018.06.19  | :---------: | :--------: |                                            | 
| | | | | | | | | 
| Embedder   | Triplet_Embedder | 1.0          |            | 2018.06.19  | :---------: | :--------: | With Batch: 1.4/person                     | 
| Embedder   | ArcFace_Embedder | 1.0          |            | 2018.06.19  | :---------: | :--------: | With Batch: 3/person ; Image:   18/person  | 
| | | | | | | | | 
| F_Identifier    | ArcFaceReid | 1.0          |            | 2018.06.19  | :---------: | :--------: |                                            |
| | | | | | | | | 
| P_Identifier    | TripletReid | 1.0          |            | 2018.06.19  | :---------: | :--------: |                                            |
| | | | | | | | | 

## Todo List
* **Skip Verification for faces**
* **GPU resource allocation**
* **Batch based embedding haven't check max size**
* Cross view tracking
* Filter out poor feature with mean and std
* **OpenPose** 
* Inter-view threshold may be calculated by tracker
* MTCNN optimize (send batch )
* MTCNN detect wrong face when inference in Yolo
* Server database access speed

## Modifications
6.19 Multi processor version

## **We are always welcome for questions and suggestions  !!**

## References
| Module             | Source                                                                                                             |
| :----------------: | :----------------------------------------------------------------------------------------------------------------: |
| Yolov2             | https://github.com/thtrieu/darkflow |
| Yolov3             | https://github.com/madhawav/YOLO3-4-Py |
| MTCNN              | https://github.com/kpzhang93/MTCNN_face_detection_alignment |
| ConvPoseMachine    | https://github.com/ildoonet/tf-pose-estimation |
| MaskRCNN           | https://github.com/matterport/Mask_RCNN |
| IOUTracker         | https://github.com/bochinski/iou-tracker |
| Triplet_Loss       | https://github.com/VisualComputingInstitute/triplet-reid |
| Arc_Face           | https://github.com/deepinsight/insightface |
| Align_Reid         | https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch |
