from scipy.stats import pearsonr as pho
from scipy.spatial.distance import euclidean as eDist
import time
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
tf.logging.set_verbosity(0)
from matplotlib import pyplot as plt
from PIL import Image
from os import path
from utils import label_map_util
from utils import visualization_utils as vis_util
import time
import cv2
from scipy import stats
import multiprocessing
from pprint import pprint

def detect_end(hsv):
    lower = np.array([135,60,60], dtype=np.uint8)
    upper = np.array([175, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    x,y = mask.nonzero()
    if len(x) < 30:
        return mask,False

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    #print ("endline slope:",slope)
    if abs(r_value) < 0.5:
        return mask,False
    return mask,True

def detect_left(hsv):
    #print ("hsv")
    #print (hsv)
    lower = np.array([23,60,60], dtype=np.uint8)
    upper = np.array([32,255,255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    x,y = mask.nonzero()
    if len(x) < 30:
        return mask,-1,-1
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    if abs(r_value) < 0.5:
        return mask,-1,-1
    return mask,slope*25+intercept,slope

def detect_right(hsv):
    sensitivity = 50
    lower = np.array([0,0,255-sensitivity], dtype=np.uint8)
    upper = np.array([180, sensitivity, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    x,y = mask.nonzero()
    if len(x) < 30:
        return mask,-1,-1

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    if abs(r_value) < 0.5:
        return mask,-1,-1  
    return mask,slope*25+intercept,slope   

def detect_left_right_end_lines(sliced_frame):
    print (sliced_frame)
    hsv = cv2.cvtColor(sliced_frame, cv2.COLOR_BGR2HSV)
    left_mask,left_point,left_slope = detect_left(hsv)
    right_mask,right_point,right_slope = detect_right(hsv)
    end_mask,end_flag = detect_end(hsv)
    return left_mask,left_point,left_slope,right_mask,right_point,right_slope,end_mask,end_flag

class vision():
    def __init__(self,cap,vis_flag=True):
        # change it to the camera for real application
        # self.cap = cv2.VideoCapture("IGVC_2015_Speed_Record.mp4")
        # self.start_time = time.time()
        self.cap = cap
        # self.cap = cv2.VideoCapture("outside3.mp4")
        # self.cap = cv2.VideoCapture("test2.mp4")
        # self.cap = cv2.VideoCapture("test_video/outside3.mp4")
        # self.cap = cv2.VideoCapture(0)
        # self.time_stamp = 0
        self.go = False
        self.magenta2m = False
        self.magenta0m = False
        self.left_lane_point= [-1]*3
        self.right_lane_point = [-1]*3
        self.endline_alert = [False]*3
        self.left_turn_flag = False
        self.right_turn_flag = False
        self.shortcut_entrance = [-1,-1]
        self.traffic_light_box = []
        self.left_mask = [-1]*3
        self.right_mask = [-1]*3
        self.end_mask = [-1]*3
        # self.slice_width = 50
        self.Y1m = 430
        self.Y2m = 350
        self.Y3m = 270
        self.Y1m = 300
        self.Y2m = 250
        self.Y3m = 200

    def start_engine(self,sess,image_tensor,detection_boxes,detection_scores,detection_classes,num_detections):
        # for traffic light detection
        ret,self.frame = self.cap.read()
        if ret == False:
            print ("Nothing read in")
            return 0
        image_cv2 = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_cv2, axis=0)
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})
        if scores[0][0] > 0.5:
            if classes[0][0] == 2:
                self.go = True
        if self.vis_flag:
            im_width=self.frame.shape[1]
            im_height = self.frame.shape[0]
            ymin, xmin, ymax, xmax = boxes[0][0].tolist()
            left, right, top, bottom = map(lambda x:int(x),[xmin * im_width, xmax * im_width,ymin * im_height, ymax * im_height])
            self.traffic_light_box = [left, right, top, bottom]   

    def detect_front_lanes(self):
        ret, self.frame = self.cap.read()
        if ret == False:
            print ("Nothing read in")
            return 0        # print (self.frame)
        # parallel to compute lines
        slice_array = [self.frame[self.Y1m:self.Y1m+50,:,:],self.frame[self.Y2m:self.Y2m+50,:,:],self.frame[self.Y3m:self.Y3m+50,:,:]]
        pool = multiprocessing.Pool(processes=3)
        #print ("slicearray:",slice_array[0])
        result_list = pool.map(detect_left_right_end_lines, slice_array)
        #result_list = [detect_left_right_end_lines(slice_array[0])]
        for i in range(3):
            left_mask,left_point,left_slope,right_mask,right_point,right_slope,end_mask,end_alert = result_list[i]
            self.left_lane_point[i] = left_point
            self.right_lane_point[i] = right_point
            self.endline_alert[i] = end_alert
            self.left_mask[i] = left_mask
            self.right_mask[i] = right_mask
            self.end_mask[i] = end_mask
        if self.left_lane_point[2] > self.left_lane_point[1] > self.left_lane_point[0]:
            self.right_turn_flag = True
        else:
            self.right_turn_flag = False
        if self.right_lane_point[2] > self.right_lane_point[1] > self.right_lane_point[0]:
            self.left_turn_flag = True
        else:
            self.left_turn_flag = False

    def left_cam_lane(self):
        ret, frame = self.cap.read()
        if ret == False:
            return 0
        



if __name__ == "__main__":

    v = vision(cv2.VideoCapture("IARRC2015_1.mp4"))

    """
    Detect traffic lights and draw bounding boxes around the traffic lights
    :param PATH_TO_TEST_IMAGES_DIR: testing image directory
    :param MODEL_NAME: name of the model used in the task
    :return: commands: True: go, False: stop
    """

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = 'traffic_inference_graph/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'traffic_light.pbtxt'

    # number of classes for COCO dataset
    NUM_CLASSES = 3

    #--------Load a (frozen) Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


    #----------Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    # out = cv2.VideoWriter('result1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))

    boxes_list = []
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('traffic-test1.avi',fourcc, 20.0, (640,480))
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')            
            while True:
                v.detect_front_lanes()
                pprint (vars(v))
                frame = v.frame.copy()
                cv2.imshow('frame',frame)
                out.write(frame)
                # out.write(frame)
                # cv2.imshow('green',v.green_mask)
                # cv2.imshow('red',v.red_mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    out.release()
    cv2.destroyAllWindows() 











