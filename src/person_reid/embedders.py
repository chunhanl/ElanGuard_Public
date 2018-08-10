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

import os
import numpy as np
import tensorflow as tf
from importlib import import_module
import cv2


class AlignReid_Embedder(Embedder):
    

    class ExtractFeature(object):
      """A function to be called in the val/test set, to extract features.
      Args:
        TVT: A callable to transfer images to specific device.
      """
    
      def __init__(self, model, TVT):
        self.model = model
        self.TVT = TVT
    
      def __call__(self, ims):
        import torch
        from torch.autograd import Variable
        old_train_eval_model = self.model.training
        # Set eval mode.
        # Force all BN layers to use global mean and variance, also disable
        # dropout.
        self.model.eval()
        ims = Variable(self.TVT(torch.from_numpy(ims).float()))
        global_feat, local_feat = self.model(ims)[:2]
        global_feat = global_feat.data.cpu().numpy()
        local_feat = local_feat.data.cpu().numpy()
        # Restore the model to its old train/eval mode.
        self.model.train(old_train_eval_model)
        return global_feat, local_feat
    
    def __init__(self,):
        os.chdir('./src/person_reid/AlignedReID-Re-Production-Pytorch-master/')
        import torch
        from torch.nn.parallel import DataParallel
        from aligned_reid.model.Model import Model
        from aligned_reid.utils.utils import load_state_dict
        from aligned_reid.utils.utils import set_devices
        os.chdir('../../../') 
        
        
        model = Model(local_conv_out_channels= 128, num_classes=751 )
        # Model wrapper
        model_w = DataParallel(model)
        weight = './model/align_reid/model_weight.pth'
        map_location = (lambda storage, loc: storage)
        sd = torch.load(weight, map_location=map_location)
        load_state_dict(model, sd)
        print('Loaded model weights from {}'.format(weight))
        TVT, TMO = set_devices((0,))

        self.featureExtractor = AlignReid_Embedder.ExtractFeature(model_w, TVT)

    def embed(self, detections):    
        person_images      = [ ((cv2.resize(det['person_img'],(128, 256))/255)[:,:,::-1]).transpose(2,0,1) for det in detections]
        if len(person_images)>0:
            global_emb, _ = self.featureExtractor(np.array(person_images))  
            for idx, det in enumerate(detections):
                det['person_embed'] = global_emb[idx]
        return detections

    def get_input_shape(self):
        return  (256, 128, 3)

class Triplet_Embedder(Embedder):
    
    def flip_augment(self, image):
        """ Returns both the original and the horizontal flip of an image. """
        images = tf.stack([image, tf.reverse(image, [1])])
        return images


    def five_crops(self, image, crop_size):
        """ Returns the central and four corner crops of `crop_size` from `image`. """
        image_size = tf.shape(image)[:2]
        crop_margin = tf.subtract(image_size, crop_size)
        assert_size = tf.assert_non_negative(
            crop_margin, message='Crop size must be smaller or equal to the image size.')
        with tf.control_dependencies([assert_size]):
            top_left = tf.floor_div(crop_margin, 2)
            bottom_right = tf.add(top_left, crop_size)
        center       = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        top_left     = image[:-crop_margin[0], :-crop_margin[1]]
        top_right    = image[:-crop_margin[0], crop_margin[1]:]
        bottom_left  = image[crop_margin[0]:, :-crop_margin[1]]
        bottom_right = image[crop_margin[0]:, crop_margin[1]:]
        return tf.stack([center, top_left, top_right, bottom_left, bottom_right])

    
    def get_input_shape(self):
        return  (288, 144, 3)
    
    def __init__(self):
        self.embedding_dim = 128
        net_input_size = (256, 128)
        pre_crop_size = (288, 144)
        self.batch_size = 64
        
        os.chdir('./src/person_reid/triplet-reid/')

        self.image_placeholder = tf.placeholder(dtype=tf.float32,shape=(None, None, None, 3))
        resize_imgs = tf.map_fn(lambda im: tf.image.resize_images(im, pre_crop_size), self.image_placeholder)
        
        # Augment the data if specified by the arguments.
        # `modifiers` is a list of strings that keeps track of which augmentations
        # have been applied, so that a human can understand it later on.
        self.modifiers = ['original']
        
        flip_imgs = tf.map_fn(self.flip_augment, resize_imgs)
        flip_imgs = tf.concat((flip_imgs[:,0],flip_imgs[:,1]),axis=0)
        self.modifiers = [o + m for m in ['', '_flip'] for o in self.modifiers]
  
        crop_imgs = tf.map_fn( lambda im: self.five_crops(im, net_input_size), flip_imgs)
        self.crop_imgs = tf.concat((crop_imgs[:,0],crop_imgs[:,1],crop_imgs[:,2],crop_imgs[:,3],crop_imgs[:,4]),axis=0)
        self.modifiers = [o + m for o in self.modifiers for m in [
            '_center', '_top_left', '_top_right', '_bottom_left', '_bottom_right']]


        # Create the model and an embedding head.
        self.model = import_module('nets.resnet_v1_50')
        self.head = import_module('heads.fc1024')
    
        self.endpoints, body_prefix = self.model.endpoints(self.crop_imgs, is_training=False)
        with tf.name_scope('head'):
            self.endpoints = self.head.head(self.endpoints, self.embedding_dim, is_training=False)
        
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        self.session  = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        # Initialize the network/load the checkpoint.
        emb_vars=[]
        emb_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v1_50')
        emb_vars+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fully_connected')
        emb_vars+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='emb')
             
        weights={}
        weights= np.load('../../../model/triplet_loss/triplet_loss.npz')['weights']
        weights= weights.all()
        
        assign_op =[]
        for var in emb_vars:
            assign_op.append(var.assign(weights[var.name]))
        self.session.run(assign_op)

        
        os.chdir('../../../')
        
    def embed(self, detections):    
        person_images = [cv2.resize(det['person_img'],(144, 288)) for det in detections]
        if len(person_images)>0:
            emb = self.session.run(self.endpoints['emb'],feed_dict={self.image_placeholder:person_images})
            if len(self.modifiers) > 1:
                # Pull out the augmentations into a separate first dimension.
                emb = np.reshape(emb,(len(person_images), len(self.modifiers), self.embedding_dim), order='F')
                # Aggregate according to the specified parameter.
                emb = np.mean(emb,axis=1)
                
            for det, e in zip(detections, emb):
                det['person_embed'] = e
        return detections


# TEST CODES
if __name__ == '__main__':

    os.chdir('../../')   
    import glob,time    
    import src.detector.detectors as detectors
    import matplotlib.pyplot as plt
    
    # Detector and Embedder
    Y_MTCNN = detectors.Yolov2()  
    
    data_fids =  glob.glob('./src/person_reid/test_images/*/*')
    data_fids.sort()
    dets = []
    for f in data_fids:
        test_img = cv2.imread(f)
        vis_img = test_img.copy()
        det = detectors.get_default_detection()
        det['person_img'] = vis_img
        dets.append(det)   
        

    A = Triplet_Embedder()
    
    s = time.time()
    A.embed(dets[:len(dets)//2])
    A.embed(dets[len(dets)//2:])
    s2 = time.time()
    emb_bchbased = [det['person_embed'] for det in dets]
    print('Pred Time:{}, {} images'.format(s2-s,len(emb_bchbased)))
    
    dis_chart = np.zeros((len(emb_bchbased),len(emb_bchbased)))
    for i in range(len(emb_bchbased)):
        for j in range(len(emb_bchbased)):
            dis_chart[i,j]= np.sqrt( np.sum( np.square(emb_bchbased[i] - emb_bchbased[j]))+1e-12 )   
    
    
    
    B = AlignReid_Embedder()
    
    s = time.time()
    B.embed(dets[:len(dets)//2])
    B.embed(dets[len(dets)//2:])
    s2 = time.time()
    emb_bchbased = [det['person_embed'] for det in dets]
    print('Pred Time:{}, {} images'.format(s2-s,len(emb_bchbased)))
    
    dis_chart2 = np.zeros((len(emb_bchbased),len(emb_bchbased)))
    for i in range(len(emb_bchbased)):
        for j in range(len(emb_bchbased)):
            dis_chart2[i,j]= np.sqrt( np.sum( np.square(emb_bchbased[i] - emb_bchbased[j]))+1e-12 )   



