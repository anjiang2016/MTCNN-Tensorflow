#coding:utf-8
import sys
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
import cv2
import os
import numpy as np
test_mode = "ONet"
thresh = [0.9, 0.6, 0.7]
min_face_size = 24
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
batch_size = [2048, 256, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
# load pnet model
if slide_window:
    PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
else:
    PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet

# load rnet model
if test_mode in ["RNet", "ONet"]:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet

# load onet model
if test_mode == "ONet":
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)
gt_imdb = []
#gt_imdb.append("35_Basketball_Basketball_35_515.jpg")
#imdb_ = dict()"
#imdb_['image'] = im_path
#imdb_['label'] = 5
#path = "/home/mingmingzhao/data_sets/1000teacher_ori_images/"
#path = "/home/mingmingzhao/closed_eye/"
path = "/home/mingmingzhao/video/teacher_images/"
#path = "lala"
for item in os.listdir(path):
    gt_imdb.append(os.path.join(path,item))

record_all=len(gt_imdb)
start_index=0
duration=100
i=0
record_c=0
while 1:
     gt_imdb_child=gt_imdb[start_index+i:start_index+i+duration]
     record_c+=duration   
     if len(gt_imdb_child)==0:
         print 'done'
         break 
     print 'next:%d,record:%d/%d'%(len(gt_imdb_child),record_c,record_all)    
     test_data = TestLoader(gt_imdb_child[0:duration])
     start_index+=duration
     #print 'gt_imdb:%d'%(len(test_data))
     #print 'gt_imdb_one:'+gt_imdb[0]
     #path = "/home/mingmingzhao/closed_eye/"
     #os.sleep(20000)
     all_boxes,landmarks = mtcnn_detector.detect_face(test_data)
     count = 0
     for imagepath in gt_imdb_child:
         face_num=len(all_boxes[count])
         if face_num>0:
             #print 'delete:%s'%(imagepath)
             if os.path.isfile(imagepath):  
                 try:  
                     os.remove(imagepath)
                     #print 'delete:%s'%(imagepath)  
                 except:  
                     print 'delete failed....'
                     pass 
         #else: 
             #print 'harddate:%s'%(imagepath)
         #print 'face_num:%d'%(len(all_boxes[count]))
             
         count = count + 1
         #writepath = ("/home/mingmingzhao/data_sets/1000teacher_ori_images_mtcnn_v1/"+os.path.basename(imagepath))
         #print 'writepath=',writepath
         #print cv2.imwrite(writepath,image)
         #cv2.imshow("lala",image)
         #cv2.waitKey(0)    
     
     '''
     for data in test_data:
         print type(data)
         for bbox in all_boxes[0]:
             print bbox
             print (int(bbox[0]),int(bbox[1]))
             cv2.rectangle(data, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
         #print data
         cv2.imshow("lala",data)
         cv2.waitKey(0)
     '''
