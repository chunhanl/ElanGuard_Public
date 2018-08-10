# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import glob

track_result=[]
track_list = glob.glob('./Results_bp/*.npz')

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] ,
                                  image[:, :, c])
    return image

def show_mask(mask):
    import matplotlib.pyplot as plt
    msk_int = 255*np.array(mask,dtype=np.uint8)*np.ones_like(mask,dtype=np.uint8)
    msk_expand  = np.expand_dims(msk_int,-1)
    msk_rgb = np.tile(msk_expand,[1,1,3])
    plt.imshow(msk_rgb)
    plt.show()
    
for tr_l in track_list:
    tr_l = track_list[2]
    tmp = np.load(tr_l)
    tmp = tmp['tr'].all()
#    for idx in range(len(tmp['bboxes'])):
#        assert len(tmp['bboxes'])==len(tmp['maskes'])
#        y1, x1, y2, x2 = tmp['bboxes'][idx]
#        msk = tmp['maskes'][idx][y1:y2,x1:x2]
#        show_mask(msk)
#        print(track_list.index(tr_l),tmp['scores'][idx])
##        cap.set(cv2.CAP_PROP_POS_FRAMES,358)
##        ret, frame = cap.read()
#        frame = tmp['frames'][idx]
#        frame = apply_mask(frame, tmp['maskes'][idx],  tmp['color'])
#        cv2.rectangle(frame,(x1, y1),(x2, y2),(0,255,0),3)
#        plt.imshow(frame)
#        plt.show()
        
    img_set=[]
    GEI_imgs =[]        
    for idx in range(len(tmp['bboxes'])):  
        y1, x1, y2, x2 = tmp['bboxes'][idx]
        msk = tmp['maskes'][idx][y1:y2,x1:x2]
        msk_rgb = 255*np.array(msk,dtype=np.uint8)*np.ones_like(msk,dtype=np.uint8)
    #    msk_rgb  = np.tile( np.expand_dims(msk_rgb,-1),[1,1,3])
        msk_rgb = cv2.resize(msk_rgb,(88,128))
        print(track_list.index(tr_l),tmp['scores'][idx])
        plt.imshow(np.array(np.tile( np.expand_dims(msk_rgb,-1),[1,1,3]),dtype=np.uint8))
        cv2.imwrite(str(idx)+'.png',np.array(np.tile( np.expand_dims(msk_rgb,-1),[1,1,3]),dtype=np.uint8))
        plt.show()
        img_set.append(msk_rgb)   
    GEI_img = np.average(img_set,axis=0)    
    GEI_img = np.array(np.tile( np.expand_dims(GEI_img,-1),[1,1,3]),dtype=np.uint8)
    plt.imshow(GEI_img)
    plt.show()     
        

import cv2
TEST_VIDEO_PATH = './video/Cam5.avi'

cap = cv2.VideoCapture(0)
cap.open(TEST_VIDEO_PATH)

cv2.namedWindow('frame',cv2.WINDOW_GUI_EXPANDED)
cv2.resizeWindow('frame', 1200,800)



sigma_l = 0
sigma_h = 0.5
sigma_iou = 0.5
t_min = 2

file_id = 0
tracks_active = []
tracks_finished = []

ret = True
while(ret):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(frame_num)
#    np.savez_compressed(open('Frame/frame_{}.pkl'.format(frame_num),'wb'), frame=frame  )

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()