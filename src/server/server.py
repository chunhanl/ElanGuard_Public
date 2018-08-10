# -*- coding: utf-8 -*-
import time
import numpy as np
import colorsys
import cv2
import queue
import multiprocessing as mp
from src.util.datatype import Tracklets
        
class Server():    
    '''
    Server Storage Formats:
        
    S_employees/S_guests['elan_id'] =
    {
        'elan_photo':
        'elan_seedface':
        'elan_seedfaceembed':
        
        'identified_tracklets':
    }

    
    '''
    
    def get_default_identity(self,):
        return {
            'elan_photo': None,
            'elan_seedface': None,
            'elan_seedfaceembed': None,
            'identified_tracklets': []
        }
    
    
    def initial_employees(self, imgpath=None, detector=None, embedder=None, load_if_exist=False):
        if load_if_exist:
            import os
            if os.path.exists(os.path.join(imgpath,'Server_employees.pkl')):
                import pickle as pkl
                with open(os.path.join(imgpath,'Server_employees.pkl'), "rb") as loadFile:
                    self.S_employees = pkl.load(loadFile)
                for key, employee in self.S_employees.items():
                    print( key + ' Loaded' )
                return
            else:
                print(os.path.join(imgpath,'Server_employees.pkl') + ' does not exist, initializing employees')
        if imgpath is not None:
            import os, glob
            imgs = glob.glob(os.path.join(imgpath,'*'))
            imgs.sort()
            for idx,pth in enumerate(imgs):   
                print(pth)
                elan_id = pth.split('/')[-1]
                print('{}/{} Initializing employee {}'.format(idx, len(imgs), elan_id))
                em_img = cv2.imread(pth)
                em_img = cv2.cvtColor(em_img, cv2.COLOR_BGR2RGB)

                #Detector
                detections = detector.predict(em_img)
                if len(detections)==0:
                    print('Warning! Can NOT find any faces in {}, please replacewith another picture.'.format(pth))
                else:    
                    if len(detections)>1:
                        print('Please make sure each img contains exactly one person: {}'.format(pth))
                        print('Randomly pick a person for employee identity (possibly worng): {}'.format(pth))
                        detections = max(detections, key=lambda x: x['face_confidence'] if x['face_confidence'] is not None else 0)
                        detections = np.expand_dims(detections,axis=0)
                    
                    if detections[0]['face_bbox'] is None:
                        print('Please make sure each img contains one face: {}'.format(pth))
                        print('No employees initialized')
                        return
                    detections = embedder.embed(detections) 
                    
                    format_identity = self.get_default_identity()
                    format_identity['elan_photo'] = em_img
                    format_identity['elan_seedface'] = detections[0]['face_img']
                    format_identity['elan_seedfaceembed'] = detections[0]['face_embed']
                    self.S_employees[elan_id] = format_identity
                
            print('{} employees initialized'.format(len(imgs)))
            import pickle as pkl
            with open(os.path.join(imgpath,'./Server_employees.pkl'), "wb") as saveFile:
                pkl.dump(self.S_employees, saveFile)
            return
        else:
            print('Imgpath/detector/embedder are required for employee list initialization')
            return
    
    
    def get_rnd_color(self,):
        h    = np.random.randint(low=0 ,high=100,size=1)/100   
        s    = np.random.randint(low=60,high=100,size=1)/100   
        v    = 1
        gen_color = [int(x*255) for x in colorsys.hsv_to_rgb(h, s, v)]
        return gen_color
    
    def __init__(self, tracker=None):
                        
        # Server storages
        self.S_tracknum_counter = 0
        self.S_tracklets = Tracklets()
        self.S_toAnalyzer_buffer = mp.Manager().Queue()
        self.S_fromAnalyzer_buffer = mp.Manager().Queue()
        self.S_employees = {}
        self.S_guests    = {}
        
        self.S_active_detections = {}
        self.S_active_tracklets  = mp.Manager().dict()
        self.num_of_active_detections_stored = 20

        # Server Modules
        self.TRACKER = tracker

        self.PRINT_WAIT_TIME        = False
        self.PRINT_TRACK_TIME       = False
        self.PRINT_UPDATE_TIME      = False
        
        
        
        
    # Upload Detection
    def upload_detections(self, cam_id, detections):
        # Only disable IOUTracking, preserver feature linking
        if len(detections)==0:
            return
                
        
        time_start= time.time()
        
        '''
        Module : Tracker
        Instruction  : Link assign_server_id based on previous frame 
        Input_format : input_detections = array of detections defined in src/detector/detectors.py
                       previous_detections = predict dets from previous frame, S_active_tracks format defined on the top
                       s_identities = identities stored in server
        Output_format: array of detections defined in src/detector/detectors.py, 
                       WITH 'assign_server_id' filled and assign_server_color' filled if tracked
        ''' 
        if cam_id in self.S_active_detections.keys():
            detections = self.TRACKER.track(input_detections=detections, previous_detections=self.S_active_detections[cam_id])
        time_track= time.time()
        if self.PRINT_TRACK_TIME: print('Server Tracking time per frame : %f'%(time_track-time_start))

        
        #Update to  S_identities if found matched server_id        
        for det in detections:
            #No exists id in S_tracklets is matched
            if det['s_tracklet_num']==-1:
                #Assign new server_id and color 
                new_s_tracknum = self.S_tracknum_counter
                new_s_color = self.get_rnd_color()
                self.S_tracknum_counter+=1
                det['s_tracklet_num']=new_s_tracknum
                det['s_tracklet_color']=new_s_color
                det['s_elan_id']= 'unidentified'
                det['e_cam_id']=cam_id
                #Add to S_tracklets
                self.S_tracklets.add_tracklet(det = det)
            #Matched exists in S_tracklets 
            else:
                match_tracklet = self.S_tracklets.get_tracklet(det['s_tracklet_num'])
                det['s_tracklet_color']=match_tracklet.color   
                det['s_elan_id']       =match_tracklet.elan_id      
                det['e_cam_id']        =cam_id
                #Update to S_tracklets
                self.S_tracklets.update_tracklet(det = det)
             
        self.S_toAnalyzer_buffer.put(detections)            
                
                
        if cam_id not in self.S_active_detections.keys():
            self.S_active_detections[cam_id] = [detections]
        else:
            self.S_active_detections[cam_id].append(detections)
            
        if cam_id not in self.S_active_tracklets.keys():    
            self.S_active_tracklets[cam_id] =[]
                    
        if len(self.S_active_detections[cam_id])>self.num_of_active_detections_stored:
            remove_dectections = self.S_active_detections[cam_id][::self.num_of_active_detections_stored]
            for dets in remove_dectections:
                for det in dets:
                    if det['s_tracklet_num'] in self.S_active_tracklets[cam_id]:
                        self.S_active_tracklets[cam_id].remove(det['s_tracklet_num'])
                        self.S_tracklets.get_tracklet(det['s_tracklet_num']).active=False
            self.S_active_detections[cam_id] = self.S_active_detections[cam_id][-self.num_of_active_detections_stored::]             
            
        for det in detections:
            if det['s_tracklet_num'] not in self.S_active_tracklets[cam_id]:
                self.S_active_tracklets[cam_id].append(det['s_tracklet_num'])  
                self.S_tracklets.get_tracklet(det['s_tracklet_num']).active=True
            
            
            
        while True:
            try:
                result = self.S_fromAnalyzer_buffer.get(block=False)
                for cmd in result:
                    if cmd[0]=='adtr_emp':
                        tracklet = self.S_tracklets[cmd[2]]
                        tracklet.elan_id=cmd[1]
                        self.S_employees[cmd[1]]['identified_tracklets'].append(tracklet)
                    elif cmd[0]=='ad_gst':
                        format_identity = self.get_default_identity()
                        self.S_guests[cmd[2]]=format_identity
                    elif cmd[0]=='rm_gst':
                        del self.S_guests[cmd[2]]
                    elif cmd[0]=='adtr_gst':
                        tracklet = self.S_tracklets[cmd[2]]
                        tracklet.elan_id=cmd[1]
                        self.S_guests[cmd[1]]['identified_tracklets'].append(tracklet)
                    elif cmd[0]=='rmtr_gst':
                        self.S_guests[cmd[1]]['identified_tracklets'].remove( self.S_tracklets[cmd[2]] )
                    else:
                        print('Unknown command from Analyzer, please check')
            except queue.Empty:
                break



            
        time_update= time.time()
        if self.PRINT_UPDATE_TIME: print('Server Updating time per frame : %f'%(time_update-time_start))

        return


    def dump_identities(self, ROOT_PATH='./dump'):
        import os
        if os.path.isdir(ROOT_PATH)==False: os.mkdir(ROOT_PATH)
        for key, employee in self.S_employees.items():
            if len(employee['identified_tracklets'])>0:      
                EMP_PATH=  os.path.join(ROOT_PATH, str(key.split('.')[0]))
                if os.path.isdir(EMP_PATH)==False: os.mkdir(EMP_PATH)
                
                for tr in employee['identified_tracklets']:
                    for idx, per in enumerate(tr.person_imgs):
                        filename = str(key)+'_'+ str(tr.tracklet_num) + '_' + str(idx)+ '.jpg'
                        SAVE_PATH = os.path.join(EMP_PATH,filename)
                        cv2.imwrite(SAVE_PATH, per[:,:,::-1])