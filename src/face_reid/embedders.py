# -*- coding: utf-8 -*-
"""
Abstract class for detectors
"""
import abc
class Embedder(abc.ABC):
       
    @abc.abstractmethod
    def embed(self):
        'Return embed features'
        return NotImplemented
    
    @abc.abstractmethod
    def get_input_shape(self):
        'Return input shape'
        return NotImplemented



from easydict import EasyDict as edict
import mxnet as mx
import numpy as np
import cv2
from skimage import transform as trans


class ArcFace_Embedder(Embedder):
        
    def get_input_shape(self):
        pass
            
    def do_flip(self, data):
        for idx in range(data.shape[0]):
            data[idx,:,:] = np.fliplr(data[idx,:,:])
        return data
            
    def __init__(self):
        modeldir = './model/insight_face/model-r50-am-lfw/model'
        gpuid = 0
        ctx = mx.gpu(gpuid)
        self.nets = []
        image_shape = [3, 112, 112]
        modeldir_=modeldir+',0'
        for model in modeldir_.split('|'):
            vec = model.split(',')
            assert len(vec)>1
            prefix = vec[0]
            epoch = int(vec[1])
            print('loading',prefix, epoch)
            net = edict()
            net.ctx = ctx
            net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
            #net.arg_params, net.aux_params = ch_dev(net.arg_params, net.aux_params, net.ctx)
            all_layers = net.sym.get_internals()
            net.sym = all_layers['fc1_output']
            net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names = None)
            net.model.bind(data_shapes=[('data', (1, 3, image_shape[1], image_shape[2]))])
            net.model.set_params(net.arg_params, net.aux_params)
            #_pp = prefix.rfind('p')+1
            #_pp = prefix[_pp:]
            #net.patch = [int(x) for x in _pp.split('_')]
            #assert len(net.patch)==5
            #print('patch', net.patch)
            self.nets.append(net)
            
    
    def align(self, detections):
        warped_images=[]
        for det in detections:
            raw_face_image = det['face_img']
            #plt.imshow(raw_face_image) 
            #plt.show()
            image_size = [112,112]
            src = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041] ], dtype=np.float32 )
            
            if image_size[1]==112:
                src[:,0] += 8.0
            
            offset = ([
                    [det['face_bbox'][0],det['face_bbox'][1]],
                    [det['face_bbox'][0],det['face_bbox'][1]],
                    [det['face_bbox'][0],det['face_bbox'][1]],
                    [det['face_bbox'][0],det['face_bbox'][1]],
                    [det['face_bbox'][0],det['face_bbox'][1]]
                    ])
            npoint= np.array(det['face_keypoint']) - np.array(offset)
            dst = npoint#.reshape( (2,5) ).T
            tform = trans.SimilarityTransform()
            tform.estimate(dst, src)
            M = tform.params[0:2,:]
            warped = cv2.warpAffine(raw_face_image,M,(image_size[1],image_size[0]), borderValue = 0.0)
            #plt.imshow(warped)   
            warped_images.append(warped)
        return warped_images
    
    def embed(self, detections):   
        det_with_face = [ det for det in detections if det['face_img'] is not None]
        if len(det_with_face)==0:
            return detections
        
        aligned_face_images = self.align(det_with_face)    
        embeds =[]
        
        # Image_based Detection time per face : 0.018270
        #        for image in aligned_face_images:
        #            image = np.transpose( image, (2,0,1) )
        #            F = None
        #            for net in self.nets:
        #                embedding = None
        #                #ppatch = net.patch
        #                for flipid in [0,1]:
        #                    _img = np.copy(image)
        #                    if flipid==1:
        #                        #plt.imshow(np.transpose( _img, (1,2,0) )[:,:,::-1])
        #                        #plt.show()    
        #                        _img = self.do_flip(_img)
        #                        #plt.imshow(np.transpose( _img, (1,2,0) )[:,:,::-1])
        #                        #plt.show() 
        #                    input_blob = np.expand_dims(_img, axis=0)
        #                    data = mx.nd.array(input_blob)
        #                    db = mx.io.DataBatch(data=(data,))
        #                    net.model.forward(db, is_train=False)
        #                    _embedding = net.model.get_outputs()[0].asnumpy().flatten()
        #                    #print(_embedding.shape)
        #                    if embedding is None:
        #                        embedding = _embedding
        #                    else:
        #                        embedding += _embedding
        #                    _norm=np.linalg.norm(embedding)
        #                    embedding /= _norm
        #                    if F is None:
        #                        F = embedding
        #                    else:
        #                        F += embedding
        #                        #F = np.concatenate((F,embedding), axis=0)
        #                    _norm=np.linalg.norm(F)
        #                    F /= _norm
        #            embeds.append(F)
        
        # Batch_based Detection time per face : 0.004155
        batch_images = []
        for image in aligned_face_images:
            image = np.transpose( image, (2,0,1) )
            for flipid in [0,1]:
                _img = np.copy(image)
                if flipid==1:
                    _img = self.do_flip(_img)
                batch_images.append(_img)
        input_blob = np.array(batch_images)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        for net in self.nets:
            net.model.forward(db, is_train=False)
            _embedding = net.model.get_outputs()[0].asnumpy()#.flatten()
            
            tmp = []
            for i in range(0,len(_embedding),2):
                mean_flip = (_embedding[i]+_embedding[i+1])/2
                _norm=np.linalg.norm(mean_flip)
                mean_flip/= _norm
                tmp.append( mean_flip )
            embeds.append(tmp)
            
        # Instead of adding up, we temporary replace with mean
        embeds = np.mean(embeds,axis=0)
        for det, emb in zip(det_with_face, embeds):
            det['face_embed'] = emb
        
        return detections


    def embed_imgs(self, images):   
            
            aligned_face_images = images
            embeds =[]
    
            # Batch_based Detection time per face : 0.004155
            batch_images = []
            for image in aligned_face_images:
                image = np.transpose( image, (2,0,1) )
                for flipid in [0,1]:
                    _img = np.copy(image)
                    if flipid==1:
                        _img = self.do_flip(_img)
                    batch_images.append(_img)
            input_blob = np.array(batch_images)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            for net in self.nets:
                net.model.forward(db, is_train=False)
                _embedding = net.model.get_outputs()[0].asnumpy()#.flatten()
                
                tmp = []
                for i in range(0,len(_embedding),2):
                    mean_flip = (_embedding[i]+_embedding[i+1])/2
                    _norm=np.linalg.norm(mean_flip)
                    mean_flip/= _norm
                    tmp.append( mean_flip )
                embeds.append(tmp)

            return embeds

# TEST CODES
if __name__ == '__main__':
    
    import time
    import matplotlib.pyplot as plt
    import glob
    import os
    os.chdir('../../')
    import src.detector.detectors as detectors
    
    # Detector and Embedder
    Y_MTCNN = detectors.Yolov2_MTCNN()     
    embed=ArcFace_Embedder()     
    
    # Load Images
    paths = glob.glob('./src/face_reid/test_images/*.jpg')
    paths.sort()
    dets = []
    for img_path in paths:
        test_img=cv2.imread(img_path)
        s = time.time()
        result_Y = Y_MTCNN.predict(test_img)
        dets.append(result_Y[0])
        e = time.time()
        print('Detection time per frame : %f'%(e-s))
        
        vis_img = test_img.copy()
        for track in result_Y:
            x1, y1, x2, y2 = track['person_bbox']
            color = np.random.randint(low=0,high=255,size=3)    
            color = (int(color[0]),int(color[1]),int(color[2]))
            cv2.rectangle(vis_img,(x1, y1), (x2, y2),color,5)
            fx1, fy1, fx2, fy2 = track['face_bbox']
            cv2.rectangle(vis_img, (x1+fx1, y1+fy1), (x1+fx2, y1+fy2), color, 5)
            for pt in track['face_keypoint']:
                cv2.circle(vis_img, (x1+pt[0], y1+pt[1]), 5, color,5 ,1)
                
        plt.imshow(vis_img[:,:,::-1])
        plt.show()        

    
    # Test Code
    s = time.time()         
    dets = embed.embed(dets)
    embed_features = [det['face_embed'] for det in dets]
    e = time.time()
    print('Detection time per face : %f'%((e-s)/len(dets)))

    dis_chart = np.zeros((len(embed_features),len(embed_features)))
    for i in range(len(embed_features)):
        for j in range(len(embed_features)):
            dis_chart[i,j]= np.sqrt( np.sum( np.square(embed_features[i] - embed_features[j]))+1e-12 )
            
    sim_chart = np.zeros((len(embed_features),len(embed_features)))
    for i in range(len(embed_features)):
        for j in range(len(embed_features)):
            sim_chart[i,j]= np.dot( embed_features[i], embed_features[j].T )
            
            
            
            
            
            
            
            
'''
        if len(detections)>0:
            have_face_indexs =[]
            input_dets =[]
            for idx,det in enumerate(detections):
                if det['face_img'] is not None:
                    have_face_indexs.append(idx)
                    input_dets.append(det)
            if len(input_dets)>0:
                emb_results = self.FACE_EMBEDDER.embed(input_dets)           
                for i,e in zip(have_face_indexs,emb_results):
                    detections[i]['face_embed'] = e

'''