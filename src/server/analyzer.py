
import time
import multiprocessing as mp
import queue
from src.util.datatype import Tracklets

def get_default_identity():
    return {
        'elan_photo': None,
        'elan_seedface': None,
        'elan_seedfaceembed': None,
        'identified_tracklets': []
    }

class Analyzer(mp.Process):
    
    def __init__(self, src_server, face_identifier, person_identifier, parent=None):
        mp.Process.__init__(self)
        self.FACE_IDENTIFIER = face_identifier
        self.PERSON_IDENTIFIER = person_identifier
        self.employees       = src_server.S_employees   
        self.guests          = {}
        self.detects_buffer  = src_server.S_toAnalyzer_buffer  
        self.output_buffer   = src_server.S_fromAnalyzer_buffer  
        self.tracklets       = Tracklets()
        # Server Embedding Charts
        self.table_pedmean_distance =[[1e-06]]
        self.PRINT_IDENTIFY_TIME    = False
        self.runnable = False
        self.assign_id = 0
        self.period = 0.1
        
        
    def run(self,):
        print('Identifier start!')
        self.runnable = True
        while self.runnable: 

            active_tracklets ={}            
            while True:
                try:
                    detect = self.detects_buffer.get(block=False)
                    for det in detect:
                        #No exists id in S_tracklets is matched
                        if det['s_tracklet_num'] not in self.tracklets.keys():
                            #Add to S_tracklets
                            self.tracklets.add_tracklet(det)
                        #Matched exists in S_tracklets 
                        else:
                            #Update to S_tracklets
                            self.tracklets.update_tracklet(det)
                        active_tracklets[det['s_tracklet_num']]=self.tracklets.get_tracklet(det['s_tracklet_num'])

                except queue.Empty:
                    break






            time_start= time.time()
            '''
            Module : Face_Identifier
            Instruction  : Verification with face
            Input_format : input_detections = array of detections defined in src/detector/detectors.py
                           s_employees = employees stored in server
            Output_format: none, the tracklet['elan_id'] will be assigned if verificated
            '''    
            f_result=[]
            if(self.FACE_IDENTIFIER is not None):
                if len(self.employees)>0: 
                    f_result = self.FACE_IDENTIFIER.identify(tracklets=active_tracklets, employees=self.employees, guests=self.guests )
            time_identify= time.time()
            if self.PRINT_IDENTIFY_TIME: print('Analyzer Face Identify time per time : %f'%(time_identify-time_start))
        
            
            '''
            Module : Person_Identifier
            Instruction  : Verification with person
            Input_format : input_detections = array of detections defined in src/detector/detectors.py
                           s_employees = employees stored in server
            Output_format: none, the tracklet['elan_id'] will be assigned if verificated
            '''        
            p_result=[]
            if(self.PERSON_IDENTIFIER is not None):
                 if len(self.employees)>0: 
                    p_result = self.PERSON_IDENTIFIER.identify(tracklets=active_tracklets, employees=self.employees, guests=self.guests )              
            time_identify2= time.time()
            if self.PRINT_IDENTIFY_TIME: print('Analyzer Person Identify time per time : %f'%(time_identify2-time_identify))



            '''
            Collect unknown Tracklets
            '''
            g_result=[]
            min_required_det = 5
            for track_num, tracklet in self.tracklets.items():
                if tracklet.elan_id=='unknown' or tracklet.elan_id=='unidentified':
#                    if tracklet.cam_id==1:
                    if len(tracklet.person_bboxs) >=min_required_det:
                        tracklet.elan_id = 'guest_{}'.format(self.assign_id)
                        format_identity = get_default_identity()
                        format_identity['identified_tracklets'] = [tracklet]
                        self.guests[tracklet.elan_id]=format_identity
                        self.assign_id+=1
                        g_result.append(('ad_gst', None, tracklet.elan_id))
                        g_result.append(('adtr_gst', tracklet.elan_id, track_num))

#                            
#            while len(self.server.S_tracklets) > len(self.table_pedmean_distance) :
#                self.table_pedmean_distance = list(np.append(self.table_pedmean_distance, np.ones( (1,len(self.table_pedmean_distance)))*-1, axis=0 ))
#                self.table_pedmean_distance = list(np.append(self.table_pedmean_distance, np.ones( (len(self.table_pedmean_distance),1))*-1, axis=1 ))
#
#            for track_num, tracklet in self.server.S_tracklets.items():
#                if tracklet.active==False and self.table_pedmean_distance[track_num][0]==-1:
#                    self.inserted_tracklet_num.append(track_num)
#                    for tr_num in self.inserted_tracklet_num:
#                        tr = self.server.S_tracklets[tr_num]
#                        dis = np.sqrt( np.sum( np.square(tracklet.person_embeds_mean -tr.person_embeds_mean))+1e-12 )
#                        self.table_pedmean_distance[track_num][tr_num] = dis




            self.output_buffer.put(f_result+p_result+g_result)
            time.sleep(self.period)
                
                

                
                
                
                
                
                
                
                
                
                
                
                
                
                
