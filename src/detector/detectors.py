# -*- coding: utf-8 -*-
"""
Detector Output format:

dets = [ detection1, detection2, detection3 ... ]
detection = 
"""

def get_default_detection():
    return {
            's_tracklet_num': -1,
            's_elan_id':'unidentified',
            's_tracklet_color': None,
            'e_cam_id': None,
            'e_frame_num' : None,
            
            
            'person_bbox' : None,
            'person_img'  : None,
            'person_seg'  : None,
            'person_embed': None,
            'person_confidence': None,
                
            'face_bbox'   : None,
            'face_img'    : None,
            'face_keypoint' : None,
            'face_embed': None,
            'face_confidence': None
    }

import time,cv2

# Abstract class for detectors
import abc
class Detector(abc.ABC):
 
    @abc.abstractmethod
    def predict(self):
        'Return detected result'
        return NotImplemented
    
    
    
class Yolov2(Detector):
    
    def __init__(self):
        from darkflow.net.build import TFNet
        import os
        MODEL_PATH = os.getcwd() +'/model/yolo/weight/yolov2.weights'
        CONFIG_PATH= os.getcwd() +"/model/yolo/cfg/yolov2.cfg"
        os.chdir('./src/detector/darkflow')
        
        options = {"model":CONFIG_PATH , "load": MODEL_PATH, "threshold": 0.1, "gpu": 0.5}
        tfnet = TFNet(options)
        os.chdir('../../../')
        self.predictor =  tfnet
        self.detect_threshold = 0.6       
        
    def predict(self, imgcv):
        yolo_result =self.predictor.return_predict(imgcv)
        copy_image = imgcv.copy()
        formatted_dets = []
        for det in yolo_result:
            if det['confidence']>self.detect_threshold and det['label']=='person':
                x1, y1, x2, y2 = det['topleft']['x'],  det['topleft']['y'], det['bottomright']['x'], det['bottomright']['y'] 
                bbox_img = copy_image[ y1:y2, x1:x2]
                
                detection = get_default_detection()
                detection['person_bbox' ] =  [x1, y1, x2, y2]
                detection['person_img' ]  =  bbox_img
                detection['person_confidence' ]  = det['confidence']
                formatted_dets.append(detection)
        return formatted_dets





class Yolov3(Detector):
    
    def __init__(self):
        import os
        MODEL_PATH = os.getcwd() +'/model/yolo/weight/yolov3.weights'
        CONFIG_PATH= os.getcwd() +"/model/yolo/cfg/yolov3.cfg"
        DATA_PATH  = os.getcwd() +"/model/yolo/cfg/coco.data"
        os.chdir('./src/detector/YOLO3-4-Py-master')    
        from pydarknet import Detector as dec
        self.in_h, self.in_w = 416,416
        self.net = dec(bytes(CONFIG_PATH, encoding="utf-8"), bytes(MODEL_PATH, encoding="utf-8"), 0, bytes(DATA_PATH, encoding="utf-8"))
        self.detect_threshold = 0.6  
        os.chdir('../../../')
        
        
    def predict(self, imgcv):  
        from pydarknet import Image
        copy_image = imgcv.copy()
        formatted_dets =[]
        origin_h, origin_w, _ = imgcv.shape
        ratio_h = origin_h/self.in_h
        ratio_w = origin_w/self.in_w
        imgcv = cv2.resize(imgcv,(self.in_w,self.in_h))
        dark_frame = Image(imgcv)
        results = self.net.detect(dark_frame)
        del dark_frame
        for cat, score, bounds in results:
            if cat==b'person' and score >=self.detect_threshold:
                x, y, w, h = bounds
                x, w = x*ratio_w, w*ratio_w
                y, h = y*ratio_h, h*ratio_h
                x1, y1= max(0,int(x-w//2)),  max(0,int(y-h//2))
                x2, y2= max(0,int(x+w//2)),  max(0,int(y+h//2))
                bbox_img = copy_image[ y1:y2, x1:x2]
                detection = get_default_detection()
                detection['person_bbox' ] =  [x1, y1, x2, y2]
                detection['person_img' ]  =  bbox_img
                detection['person_confidence' ]  = score
                formatted_dets.append(detection)
        
        return formatted_dets


class MTCNN(Detector):
    def __init__(self):
        import os
        os.chdir('./src/detector/')
        from MTCNN import detect_face 
        os.chdir('../../')
        import tensorflow as tf
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, './model/mtcnn/')

        self.minsize = 50 # minimum size of face
        self.threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        self.factor = 0.709 # scale factor

        
    def predict(self, imgcv):
        import sys
        sys.path.append('./src/detector/')
        from MTCNN import detect_face 
        (self.face_bbox, self.face_organs) = detect_face.detect_face(imgcv, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        formatted_dets =  []
        if len(self.face_bbox)>0:
            for f_bbox, f_org in zip(self.face_bbox, self.face_organs.T):
                detection = get_default_detection()
                x1, y1, x2, y2 = f_bbox[:4]
                x1, y1, x2, y2 = max(0,x1), max(0,y1), max(0,x2), max(0,y2)
                face_img = imgcv[int(y1):int(y2),int(x1):int(x2)]
                detection['face_img']  = face_img
                detection['face_bbox'] = [int(x1), int(y1), int(x2), int(y2)]
                detection['face_confidence'] = f_bbox[4]
                detection['face_keypoint'] = [[int(f_org[0]),int(f_org[5])],
                                              [int(f_org[1]),int(f_org[6])], 
                                              [int(f_org[2]),int(f_org[7])],
                                              [int(f_org[3]),int(f_org[8])], 
                                              [int(f_org[4]),int(f_org[9])] ]
                formatted_dets.append(detection)
        return formatted_dets
    

class MTCNN_Batch(Detector):
    def __init__(self):
        import os
        os.chdir('./src/detector/')
        from MTCNN import detect_face 
        os.chdir('../../')
        import tensorflow as tf
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, './model/mtcnn/')

        self.minsize_ratio = 0.05
        self.threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        self.factor = 0.709 # scale factor

        
    def predict(self, imgcvs):
        import sys
        sys.path.append('./src/detector/')
        from MTCNN import detect_face 
        results = detect_face.bulk_detect_face(imgcvs, self.minsize_ratio, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        self.aa = results
        all_dets = []
        for idx in range(0, len(results)):
            formatted_dets =  []
            if results[idx] is not None:
                (self.face_bbox, self.face_organs)= results[idx]
                if len(self.face_bbox)>0:
                    for f_bbox, f_org in zip(self.face_bbox, self.face_organs.T):
                        detection = get_default_detection()
                        x1, y1, x2, y2 = f_bbox[:4]
                        x1, y1, x2, y2 = max(0,x1), max(0,y1), max(0,x2), max(0,y2)
                        face_img = imgcvs[idx][int(y1):int(y2),int(x1):int(x2)]
                        detection['face_img']  = face_img
                        detection['face_bbox'] = [int(x1), int(y1), int(x2), int(y2)]
                        detection['face_confidence'] = f_bbox[4]
                        detection['face_keypoint'] = [[int(f_org[0]),int(f_org[5])],
                                                      [int(f_org[1]),int(f_org[6])], 
                                                      [int(f_org[2]),int(f_org[7])],
                                                      [int(f_org[3]),int(f_org[8])], 
                                                      [int(f_org[4]),int(f_org[9])] ]
                        formatted_dets.append(detection)
            all_dets.append(formatted_dets)
        return all_dets

class MTCNN_Batch_FixedSize(Detector):
    def __init__(self):
        import os
        os.chdir('./src/detector/')
        from MTCNN import detect_face 
        os.chdir('../../')
        import tensorflow as tf
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, './model/mtcnn/')

        self.minsize_ratio = 0.1
        self.threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        self.factor = 0.709 # scale factor
        
        self.fixed_h = 350
        
    def predict(self, imgcvs):
        import sys
        sys.path.append('./src/detector/')
        from MTCNN import detect_face 
        ratios = [(im.shape[0]/self.fixed_h, im.shape[0]/self.fixed_h) for im in imgcvs]
        self.ratios = ratios
        self.imgs   = [cv2.resize(im, ( int(im.shape[1] * self.fixed_h/im.shape[0]), self.fixed_h)) for im in imgcvs]
        results = detect_face.bulk_detect_face(self.imgs, self.minsize_ratio, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        all_dets = []
        for idx in range(0, len(results)):
            formatted_dets =  []
            if results[idx] is not None:
                (self.face_bbox, self.face_organs)= results[idx]
                if len(self.face_bbox)>0:
                    for f_bbox, f_org in zip(self.face_bbox, self.face_organs.T):
                        detection = get_default_detection()
                        x1, y1, x2, y2 = f_bbox[:4]
                        x1, y1, x2, y2 = max(0,x1), max(0,y1), max(0,x2), max(0,y2)
                        x1, y1, x2, y2 = x1*ratios[idx][1], y1*ratios[idx][0], x2*ratios[idx][1], y2*ratios[idx][0]
                        face_img = imgcvs[idx][int(y1):int(y2),int(x1):int(x2)]
                        detection['face_img']  = face_img
                        detection['face_bbox'] = [int(x1), int(y1), int(x2), int(y2)]
                        detection['face_confidence'] = f_bbox[4]
                        detection['face_keypoint'] = [[int(f_org[0]*ratios[idx][1]),int(f_org[5]*ratios[idx][0])],
                                                      [int(f_org[1]*ratios[idx][1]),int(f_org[6]*ratios[idx][0])], 
                                                      [int(f_org[2]*ratios[idx][1]),int(f_org[7]*ratios[idx][0])],
                                                      [int(f_org[3]*ratios[idx][1]),int(f_org[8]*ratios[idx][0])], 
                                                      [int(f_org[4]*ratios[idx][1]),int(f_org[9]*ratios[idx][0])] ]
                        formatted_dets.append(detection)
            all_dets.append(formatted_dets)
        return all_dets  

class Yolov2_MTCNN(Detector):
   
    def __init__(self):
        self.yolo  = Yolov2()
        self.mtcnn = MTCNN()

    def predict(self, imgcv):
        formatted_dets =  self.yolo.predict(imgcv)
        for det in formatted_dets:
            mtcnn_dets= self.mtcnn.predict(det['person_img'])
            if len(mtcnn_dets)>0:
                mtcnn_det = max(mtcnn_dets, key=lambda x: x['face_confidence'])
                det['face_img']  = mtcnn_det['face_img']
                det['face_bbox'] = mtcnn_det['face_bbox']
                det['face_confidence'] = mtcnn_det['face_confidence']
                det['face_keypoint']   = mtcnn_det['face_keypoint']
        return formatted_dets
    
    
class Yolov3_MTCNN(Detector):
   
    def __init__(self):
        self.yolo  = Yolov3()
        self.mtcnn = MTCNN()

    def predict(self, imgcv):
        formatted_dets =  self.yolo.predict(imgcv)
        for det in formatted_dets:
            mtcnn_dets= self.mtcnn.predict(det['person_img'])
            if len(mtcnn_dets)>0:
                mtcnn_det = max(mtcnn_dets, key=lambda x: x['face_confidence'])
                det['face_img']  = mtcnn_det['face_img']
                det['face_bbox'] = mtcnn_det['face_bbox']
                det['face_confidence'] = mtcnn_det['face_confidence']
                det['face_keypoint']   = mtcnn_det['face_keypoint']
        return formatted_dets

class Yolov3_MTCNN_Batch(Detector):
   
    def __init__(self):
        self.yolo  = Yolov3()
        self.mtcnn = MTCNN_Batch_FixedSize()

    def predict(self, imgcv):

        formatted_dets =  self.yolo.predict(imgcv)

        imgs = [det['person_img'] for det in formatted_dets]
        results = self.mtcnn.predict(imgs)
        for res,det in zip(results, formatted_dets):
            if len(res)>0:
                mtcnn_det = max(res, key=lambda x: x['face_confidence'])
                det['face_img']  = mtcnn_det['face_img']
                det['face_bbox'] = mtcnn_det['face_bbox']
                det['face_confidence'] = mtcnn_det['face_confidence']
                det['face_keypoint']   = mtcnn_det['face_keypoint']

        
        return formatted_dets

class MaskRCNN(Detector):
    
    def __init__(self):    
        import os
        import sys
        # Root directory of the project
        ROOT_DIR = os.path.abspath("./src/detector/Mask_RCNN")
        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        import mrcnn.model as modellib
        # Import COCO config
        sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
        import coco

        
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join('model', 'mask_rcnn',"mask_rcnn_coco.h5")
        
        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        
        config = InferenceConfig()
        config.POST_NMS_ROIS_INFERENCE = 50
        config.POST_NMS_ROIS_TRAINING  = 50
        config.display()
        
        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        
        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)
        
        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']
        self.predictor =  model  
        self.detect_threshold = 0.6
        
        import cv2
        test_img = cv2.imread('./src/detector/detectors_test_image.jpg')
        
        if test_img is not None:
            print('Warming up detector MaskRCNN....')
            s = time.time()
            self.predict(test_img)
            e = time.time()
            print('MaskRCNN Warmed, Time: {}'.format(e-s))
        else:
            print('Test img missing, skip warming up detector MaskRCNN....')
            print('The first detection will take longer time for MaskRCNN....')
    def predict(self, imgcv):
        mask_result = self.predictor.detect([imgcv], verbose=0)
        copy_image = imgcv.copy()
        formatted_dets = []
        for i in range( len(mask_result[0]['rois'])):
            if mask_result[0]['scores'][i] >= self.detect_threshold and mask_result[0]['class_ids'][i]== 1:
                x1, y1, x2, y2 = mask_result[0]['rois'][i,1], mask_result[0]['rois'][i,0], mask_result[0]['rois'][i,3], mask_result[0]['rois'][i,2]
                bbox_img = copy_image[ y1:y2, x1:x2]
                
                detection = get_default_detection()
                detection['person_bbox'] =  [x1, y1, x2, y2]
                detection['person_img']  =  bbox_img
                detection['person_seg']  = mask_result[0]['masks'][y1:y2, x1:x2, i]
                detection['person_confidence']  = mask_result[0]['scores'][i]
                formatted_dets.append(detection)

        return formatted_dets
         

class ConvPoseMachine(Detector):
    
    def __init__(self, input_wicth=800 , input_height=640):
        import sys
        sys.path.append('./src/detector/tf-pose-estimation/src')       

        self.model = 'cmu'
        import common 
        self.enum_coco_parts = common.CocoPart
        self.enum_coco_colors= common.CocoColors      
        self.enum_coco_pairs_render= common.CocoPairsRender
        
        from estimator import TfPoseEstimator
        from networks import get_graph_path
        self.image_h, self.image_w = input_height, input_wicth
        if self.image_w % 16 != 0 or self.image_h % 16 != 0:
                raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (self.image_w, self.image_h))
        print('Warming up detector ConvPoseMachine....')
        import time 
        s = time.time()
        self.estimator = TfPoseEstimator(get_graph_path(self.model), target_size=(self.image_w, self.image_h))  
        e = time.time()
        print('ConvPoseMachine Warmed, Time: {}'.format(e-s))


    def predict(self, imgcv):
        # Build model based on input image size
        img_h, img_w, _ =  imgcv.shape   
        
        humans = self.estimator.inference(imgcv)
        formatted_dets = []

        for human in humans:
            key_point = {}
            # draw point
            for i in range(self.enum_coco_parts.Background.value):
                if i not in human.body_parts.keys():
                    continue
    
                body_part = human.body_parts[i]
                x = int((body_part.x * self.image_w + 0.5)* img_w/self.image_w)
                y = int((body_part.y * self.image_h + 0.5)* img_h/self.image_h)
                center = (x,y)
                
                key_point[i] = center
       
            detection = get_default_detection()
            detection['person_keypoint'] = key_point
            formatted_dets.append(detection)
    
        return formatted_dets
    
    
class Yolov2_CPM(Yolov2):

    def __init__(self):
        super(Yolov2_CPM, self).__init__()
        self.CPM = ConvPoseMachine(input_wicth=400 , input_height=880)

                
    def predict(self, imgcv):
        formatted_dets =  super(Yolov2_CPM, self).predict(imgcv)
        for det in formatted_dets:
            humans = self.CPM.predict(det['person_img'])
            human = humans[0]                
            det['person_keypoint'] = human['person_keypoint']
        return formatted_dets







# TEST CODES
if __name__ == '__main__':
    import os
    import cv2
    import matplotlib.pyplot as plt
    import time
    import numpy as np  
########################################Test Image#############################
    test_img = cv2.imread('detectors_test_image2.jpg')
#    test_img = cv2.resize(test_img,(1200,800))
    os.chdir('../../')
    plt.imshow(test_img[:,:,::-1])
    plt.show()
    
    
#########################################Yolo v2 ###############################
#    Y2 = Yolov2()
#    s = time.time()
#    result_Y = Y2.predict(test_img)
#    e = time.time()
#    print('Detection time per frame : %f'%(e-s))
#    
#    vis_img = test_img.copy()
#    for track in result_Y:
#        x1, y1, x2, y2 = track['person_bbox']
#        color = np.random.randint(low=0,high=255,size=3)    
#        color = (int(color[0]),int(color[1]),int(color[2]))
#        cv2.rectangle(vis_img,(x1, y1), (x2, y2),color,5)
#    plt.imshow(vis_img[:,:,::-1])
#    plt.show()
#    
######################################Yolov3####################################
#    Y3 = Yolov3()
#    
#    s = time.time()
#    result_Y = Y3.predict(test_img)
#    e = time.time()
#    print('Detection time per frame : %f'%(e-s))
#    
#    vis_img = test_img.copy()
#    for track in result_Y:
#        x1, y1, x2, y2 = track['person_bbox']
#        color = np.random.randint(low=0,high=255,size=3)    
#        color = (int(color[0]),int(color[1]),int(color[2]))
#        cv2.rectangle(vis_img,(x1, y1), (x2, y2),color,5)
#    plt.imshow(vis_img[:,:,::-1])
#    plt.show()
    
    
#########################################MaskRCNN###############################
#    M = MaskRCNN()
#    s = time.time()
#    result_M = M.predict(test_img)
#    e = time.time()
#    print('Detection time per frame : %f'%(e-s))
#    
#    vis_img = test_img.copy()
#    for track in result_M:
#        x1, y1, x2, y2 = track['person_bbox']
#        color = np.random.randint(low=0,high=255,size=3)    
#        color = (int(color[0]),int(color[1]),int(color[2]))
#        cv2.rectangle(vis_img,(x1, y1), (x2, y2),color,5)
#        
#        alpha = 0.5
#        for c in range(3):
#            vis_img[y1:y2, x1:x2, c] = np.where(track['person_seg'] == 1,
#                                      vis_img[y1:y2, x1:x2, c] *
#                                      (1 - alpha) + alpha * color[c] ,
#                                      vis_img[y1:y2, x1:x2, c])
    
    
####################################MTCNN############################
#    M = MTCNN()
#    
#    s = time.time()
#    result = M.predict(test_img)
#    e = time.time()
#    print('Detection time per frame : %f'%(e-s))
#    
#    vis_img = test_img.copy()
#    for track in result:
#        x1, y1, x2, y2 = 0,0,0,0
#        color = np.random.randint(low=0,high=255,size=3)    
#        color = (int(color[0]),int(color[1]),int(color[2]))
#        if track['face_bbox'] is not None:
#            fx1, fy1, fx2, fy2 = track['face_bbox']
#            cv2.rectangle(vis_img, (x1+fx1, y1+fy1), (x1+fx2, y1+fy2), color, 5)
#            for pt in track['face_keypoint']:
#                cv2.circle(vis_img, (x1+pt[0], y1+pt[1]), 5, color,5 ,1)
#                
#    vis_img = cv2.resize(vis_img,(1200,800))
#    plt.imshow(vis_img[:,:,::-1])
#    plt.show()
#    cv2.imshow('frame',vis_img)
#    if cv2.waitKey():
#        cv2.destroyAllWindows()
#    

######################################Yolov2 + MTCNN############################
#    Y2_MTCNN = Yolov2_MTCNN()
#    
#    s = time.time()
#    result_Y = Y2_MTCNN.predict(test_img)
#    e = time.time()
#    print('Detection time per frame : %f'%(e-s))
#    print('Detection time per person : %f'%((e-s)/len(result_Y)))
#    
#    vis_img = test_img.copy()
#    for track in result_Y:
#        x1, y1, x2, y2 = track['person_bbox']
#        color = np.random.randint(low=0,high=255,size=3)    
#        color = (int(color[0]),int(color[1]),int(color[2]))
#        cv2.rectangle(vis_img,(x1, y1), (x2, y2),color,5)
#        if track['face_bbox'] is not None:
#            fx1, fy1, fx2, fy2 = track['face_bbox']
#            cv2.rectangle(vis_img, (x1+fx1, y1+fy1), (x1+fx2, y1+fy2), color, 5)
#            for pt in track['face_keypoint']:
#                cv2.circle(vis_img, (x1+pt[0], y1+pt[1]), 5, color,5 ,1)
#                
#    vis_img = cv2.resize(vis_img,(1200,800))
#    plt.imshow(vis_img[:,:,::-1])
#    plt.show()
#    cv2.imshow('frame',vis_img)
#    if cv2.waitKey():
#        cv2.destroyAllWindows()
#
#
#####################################Yolov3 + MTCNN############################
#    Y3_MTCNN = Yolov3_MTCNN()
#    
#    s = time.time()
#    result_Y = Y3_MTCNN.predict(test_img)
#    e = time.time()
#    print('Detection time per frame : %f'%(e-s))
#    print('Detection time per person : %f'%((e-s)/len(result_Y)))
#    
#    vis_img = test_img.copy()
#    for track in result_Y:
#        x1, y1, x2, y2 = track['person_bbox']
#        color = np.random.randint(low=0,high=255,size=3)    
#        color = (int(color[0]),int(color[1]),int(color[2]))
#        cv2.rectangle(vis_img,(x1, y1), (x2, y2),color,5)
#        if track['face_bbox'] is not None:
#            fx1, fy1, fx2, fy2 = track['face_bbox']
#            cv2.rectangle(vis_img, (x1+fx1, y1+fy1), (x1+fx2, y1+fy2), color, 5)
#            for pt in track['face_keypoint']:
#                cv2.circle(vis_img, (x1+pt[0], y1+pt[1]), 5, color,5 ,1)
#            
#    vis_img = cv2.resize(vis_img,(1200,800))
#    plt.imshow(vis_img[:,:,::-1])
#    plt.show()
#    cv2.imshow('frame',vis_img)
#    if cv2.waitKey():
#        cv2.destroyAllWindows()
#
    
####################################Yolov3 + MTCNN############################
    Y3_MTCNN = Yolov3_MTCNN_Batch()
    
    s = time.time()
    result_Y = Y3_MTCNN.predict(test_img)
    e = time.time()
    print('Detection time per frame : %f'%(e-s))
    print('Detection time per person : %f'%((e-s)/len(result_Y)))
    
    vis_img = test_img.copy()
    for track in result_Y:
        x1, y1, x2, y2 = track['person_bbox']
        color = np.random.randint(low=0,high=255,size=3)    
        color = (int(color[0]),int(color[1]),int(color[2]))
        cv2.rectangle(vis_img,(x1, y1), (x2, y2),color,5)
        if track['face_bbox'] is not None:
            fx1, fy1, fx2, fy2 = track['face_bbox']
            cv2.rectangle(vis_img, (x1+fx1, y1+fy1), (x1+fx2, y1+fy2), color, 5)
            for pt in track['face_keypoint']:
                cv2.circle(vis_img, (x1+pt[0], y1+pt[1]), 5, color,5 ,1)
            
    vis_img = cv2.resize(vis_img,(1200,800))
    plt.imshow(vis_img[:,:,::-1])
    plt.show()
    cv2.imshow('frame',vis_img)
    if cv2.waitKey():
        cv2.destroyAllWindows()


#######################Convolutional Pose Machine###############################
#    CPM = ConvPoseMachine(input_wicth=1200 , input_height=800)
#
#    s = time.time()
#    result_CPM = CPM.predict(test_img)
#    e = time.time()
#    print('Detection time per frame : %f'%(e-s))
#    
#    vis_img = test_img.copy()    
#
#    for track in result_CPM:
#        key_point = track['person_keypoint']
#        # draw point
#        for i in range(CPM.enum_coco_parts.Background.value):
#            if i not in key_point.keys():
#                continue
#            cv2.circle(vis_img, key_point[i], 3, CPM.enum_coco_colors[i], thickness=3, lineType=8, shift=0)
#        # draw line
#        for pair_order, pair in enumerate(CPM.enum_coco_pairs_render):
#            if pair[0] not in key_point.keys() or pair[1] not in key_point.keys():
#                continue
#            cv2.line(vis_img, key_point[pair[0]], key_point[pair[1]], CPM.enum_coco_colors[pair_order], 3)
#                
#    plt.imshow(vis_img[:,:,::-1])
#    plt.show()
#    
#    
#    
#######################Yolov2 + CPM#############################################    
#    YCPM = Yolov2_CPM()
#    
#    s = time.time()
#    result_YCPM = YCPM.predict(test_img)
#    e = time.time()
#    print('Detection time per frame : %f'%(e-s))
#    
#    vis_img = test_img.copy()
#
#    for track in result_YCPM:
#        # YOLO bbox
#        x1, y1, x2, y2 = track['person_bbox']
#        color = np.random.randint(low=0,high=255,size=3)    
#        color = (int(color[0]),int(color[1]),int(color[2]))
#        cv2.rectangle(vis_img,(x1, y1), (x2, y2),color,5)
#        
#        # CPM
#        key_point = track['person_keypoint']
#        # draw point
#        for i in range(YCPM.CPM.enum_coco_parts.Background.value):
#            if i not in key_point.keys():
#                continue
#            cv2.circle(vis_img[y1:y2,x1:x2], key_point[i], 3, YCPM.CPM.enum_coco_colors[i], thickness=3, lineType=8, shift=0)
#        # draw line
#        for pair_order, pair in enumerate(YCPM.CPM.enum_coco_pairs_render):
#            if pair[0] not in key_point.keys() or pair[1] not in key_point.keys():
#                continue
#            cv2.line(vis_img[y1:y2,x1:x2], key_point[pair[0]], key_point[pair[1]], YCPM.CPM.enum_coco_colors[pair_order], 3)
#        
#    plt.imshow(vis_img[:,:,::-1])
#    plt.show()
#
