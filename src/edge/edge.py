# -*- coding: utf-8 -*-
import time 
class Edge():    
    
    PRINT_PREPROCESS_TIME  = False        
    PRINT_DETECT_TIME      = False
    PRINT_FILTER_TIME      = False
    PRINT_ALIGN_TIME       = False
    PRINT_EMBED_TIME       = False
    PRINT_UPLOAD_TIME      = False
    PRINT_TOTAL_TIME       = False
    
    def __init__(self, 
                 server=None, 
                 detector=None, 
                 person_filter=None, 
                 person_aligner=None, 
                 person_embedder=None, 
                 face_filter=None, 
                 face_aligner=None, 
                 face_embedder=None, 
                 parent = None ):

        self.SERVER = server        
        self.DETECTOR = detector
        self.FACE_FILTER   = face_filter
        self.FACE_ALIGNER  = face_aligner
        self.FACE_EMBEDDER = face_embedder
        self.PERSON_FILTER   = person_filter
        self.PERSON_ALIGNER  = person_aligner
        self.PERSON_EMBEDDER = person_embedder
                      
    def pack_features(self, input_frame, cam_id=None, frame_no=None):
        time_start= time.time()
        input_copy = input_frame.copy()
        detections = []
        '''
        Module : Detector
        Input_format : uint8 image with shape self.width x self.height x self.channel
        Output_format: array of detections defined in src/detector/detectors.py
        '''
        if self.DETECTOR is not None:
            detections = self.DETECTOR.predict(input_copy)
            # Print frame number into detections if provided
            if frame_no is not None:
                for det in detections:
                    det['e_frame_num']=frame_no
        time_detect= time.time()
        if Edge.PRINT_DETECT_TIME: print('Detection time per frame : %f'%(time_detect-time_start))
        

        '''
        Module : Face_ReID_Filter
        Input_format : array of detections defined in src/detector/detectors.py
        Output_format: array of detections defined in src/detector/detectors.py, 
                       WITH unqualified 'face_img' = None
        ''' 
        if self.FACE_FILTER is not None:
            detections = self.FACE_FILTER.filter_out(detections, visual_only=True)
        time_filter= time.time()
        if Edge.PRINT_FILTER_TIME: print('Face Filter time per frame : %f'%(time_filter-time_detect))
        '''
        Module : Face_ReID_Aligner
        Input_format : array of detections defined in src/detector/detectors.py
        Output_format: array of detections defined in src/detector/detectors.py, 
                       WITH 'face_img_aligned' filled if 'face_img' is not None
        ''' 
        if self.FACE_ALIGNER is not None:
            detections = self.FACE_ALIGNER.align(detections)
        time_align= time.time()
        if Edge.PRINT_ALIGN_TIME: print('Face Align time per frame : %f'%(time_align-time_filter))
        '''
        Module : Face_ReID_Embedder
        Input_format : array of detections defined in src/detector/detectors.py
        Output_format: array of detections defined in src/detector/detectors.py, 
                       WITH 'face_embed' filled if 'face_img' is not None
        ''' 
        if self.FACE_EMBEDDER is not None:
            detections = self.FACE_EMBEDDER.embed(detections)
        time_embed= time.time()
        if Edge.PRINT_EMBED_TIME: print('Face Embedding time per frame : %f'%(time_embed-time_align))
  
        '''
        Module : Person_ReID_Filter
        Input_format : array of detections defined in src/detector/detectors.py
        Output_format: array of detections defined in src/detector/detectors.py, 
                       WITH unqualified 'person_img' = None
        ''' 
        if self.PERSON_FILTER is not None:        
            detections = self.PERSON_FILTER.filter_out(detections)#, visual_only=True)
        time_filter2= time.time()
        if Edge.PRINT_FILTER_TIME: print('Person Filter time per frame : %f'%(time_filter2-time_embed))
        '''
        Module : Person_ReID_Aligner
        Input_format : array of detections defined in src/detector/detectors.py
        Output_format: array of detections defined in src/detector/detectors.py, 
                       WITH 'person_img_aligned' filled if 'person_img' is not None
        ''' 
        if self.PERSON_ALIGNER is not None:        
            detections = self.PERSON_ALIGNER.align(detections)
        time_align2= time.time()
        if Edge.PRINT_ALIGN_TIME: print('Person Align time per frame : %f'%(time_align2-time_filter2))
        '''
        Module : Person_ReID_Embedder
        Input_format : array of detections defined in src/detector/detectors.py
        Output_format: array of detections defined in src/detector/detectors.py, 
                       WITH 'person_embed' filled if 'person_img' is not None
        ''' 
        if self.PERSON_EMBEDDER is not None:  
            detections = self.PERSON_EMBEDDER.embed(detections)
        time_emb2= time.time()
        if Edge.PRINT_EMBED_TIME: print('Person Embedding time per frame : %f'%(time_emb2-time_align2))

        '''
        Packing and Send to Server
        '''
        if self.SERVER is not None: 
            self.SERVER.upload_detections(cam_id, detections)
        time_upload= time.time()
        if Edge.PRINT_UPLOAD_TIME: print('Uploading time per frame : %f'%(time_upload-time_emb2))
        
                
        if Edge.PRINT_TOTAL_TIME: print('Total time per frame : %f'%(time_upload-time_start))
        
        return  detections
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        