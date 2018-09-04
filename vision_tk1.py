from scipy.stats import pearsonr as pho
from scipy.spatial.distance import euclidean as eDist
import time
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
# import tensorflow as tf
# tf.logging.set_verbosity(0)
from matplotlib import pyplot as plt
# from PIL import Image
from os import path
# from utils import label_map_util
# from utils import visualization_utils as vis_util
import time
import cv2


def traffic_signal_detection(frame,top,bottom,left,right,num=3,threshold=0.2):
    # if both red/green fail to pass the threshold, that means no traffic light is on, return None
    # if it has both red and green signal, something is wrong, return stop
#     ff = frame[top:bottom,left:right,:]
    height = bottom - top
    step_size = int(height/num)
    red_signal_part = frame[top:top+step_size,left:right,:]
    green_signal_part = frame[bottom-step_size:bottom,left:right,:]    
    green_flag = detect_green(green_signal_part)
    red_flag = detect_red(red_signal_part)
    if red_flag > threshold and red_flag > green_flag:
        return False
    if green_flag > threshold and green_flag > red_flag:
        return True
    return None


def cv2_traffic_light(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # REDDDDD
    lower_red = np.array([166,84,141])
    upper_red = np.array([186,255,255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)
    
    # Yellowwww
    # lower_yellow = np.array([22,60,119])
    # upper_yellow = np.array([47, 255, 255])
    # masky = cv2.inRange(hsv,lower_yellow,upper_yellow)
    
    # lower_y = np.array([0,99,99])
    # upper_y = np.array([10,255,255])
    # maskyy = cv2.inRange(hsv, lower_y, upper_y)
    
    # lower_y2 = np.array([66,166, 181])
    # upper_y2 = np.array([107,183,189])
    # masky2 = cv2.inRange(hsv, lower_y2, upper_y2)
    
    # Greennnnn
    lower_green = np.array([60, 60, 60])
    upper_green = np.array([80, 255, 255])
    maskg = cv2.inRange(hsv,lower_green,upper_green)

    # join
    kernel = np.ones((5,5), np.uint8)
    
    # RED
    mask = mask0
    mask = cv2.dilate(mask, kernel)
    # res = cv2.bitwise_and(frame, frame, mask = mask)
    # Yellow
    # mask_y = masky + maskyy + masky2
    # mask_y = cv2.dilate(mask_y, kernel)
    # resy = cv2.bitwise_and(frame, frame, mask = mask_y)
    # Green
    maskg = cv2.dilate(maskg, kernel)
    # resg = cv2.bitwise_and(frame, frame, mask = maskg)
    
    # res_final = res + resy + resg
    
    # tracking the Red color
    go_flag = False
    green_box = []
    red_box = []
    # tracking the Green color
    (_, contours,hierarchy) = cv2.findContours(maskg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 600):
                go_flag = True
                # x,y,w,h = cv2.boundingRect(contour)
                green_box = cv2.boundingRect(contour)
                # image = cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 0),2)
                # cv2.putText(frame,"Green",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),2)    
    (_, contours,hierarchy) = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
                go_flag = False
                # x,y,w,h = cv2.boundingRect(contour)
                red_box = cv2.boundingRect(contour)
                # image = cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255),2)
                # cv2.putText(frame,"Red",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)
    return  go_flag,green_box,red_box  
  

              




def detect_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower = np.array([34,60,60], dtype=np.uint8)
    upper = np.array([84, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    rate = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
    return rate

def traffic_light_extraction(frame,boxes, scores, classes):
    go_flag = False
    im_width=frame.shape[1]
    im_height = frame.shape[0]
    boxes_list = []    
    for i in range(len(classes[0])):
        if not scores[0][i] > 0.5:
            continue
        if not classes[0][i] == 10:
            continue
        ymin, xmin, ymax, xmax = boxes[0][i].tolist()
        left, right, top, bottom = map(lambda x:int(x),[xmin * im_width, xmax * im_width,ymin * im_height, ymax * im_height])
        boxes_list.append([left, right, top, bottom])
        if traffic_signal_detection(frame,top,bottom,left,right,num=3,threshold=0.2):
            go_flag = True
    return go_flag,boxes_list  

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)




def read_traffic_lights(image, boxes, scores, classes, max_boxes_to_draw=20, min_score_thresh=0.5, traffic_ligth_label=10):
    im_width, im_height = image.size
    red_flag = False
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores[i] > min_score_thresh and classes[i] == traffic_ligth_label:
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            crop_img = image.crop((left, top, right, bottom))

            if detect_red(crop_img):
                red_flag = True

    return red_flag


def plot_origin_image(image_np, boxes, classes, scores, category_index):

    # Size of the output images.
    IMAGE_SIZE = (12, 8)
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category_index,
      min_score_thresh=.5,
      use_normalized_coordinates=True,
      line_thickness=3)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)

    # save augmented images into hard drive
    # plt.savefig( 'output_images/ouput_' + str(idx) +'.png')
    plt.show()


def detect_traffic_lights(image_cv2,sess,detection_graph):

    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    # image = Image.fromarray(image_cv2)
    # image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_cv2, axis=0)
    # print (image_np_expanded.shape)
    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
      [detection_boxes, detection_scores, detection_classes, num_detections],
      feed_dict={image_tensor: image_np_expanded})

    go_flag,boxes_list = traffic_light_extraction(image_cv2,boxes, scores, classes)

    return go_flag,boxes_list

class vision():
    def __init__(self):
        # change it to the camera for real application
        # self.cap = cv2.VideoCapture("IGVC_2015_Speed_Record.mp4")
        self.start_time = time.time()
        self.cap = cv2.VideoCapture("endline2.mp4")
        self.time_stamp = 0
        self.go = False
        self.magenta2m = False
        self.magenta0m = False




    def start_engine(self,sess=None,detection_graph=None):
        # for traffic light detection
        ret,self.frame = self.cap.read()
        # self.frame = cv2.resize(self.frame,(300,300))
        if ret == False:
            print ("Nothing read in")
            return 0
        # frame = cv2.imread("test_images/test3.jpg")
        self.go,self.green_box,self.red_box = cv2_traffic_light(self.frame)
        # self.go,self.boxes_list = detect_traffic_lights(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB),sess,detection_graph)
        # self.go = detect_traffic_lights("test_images/test3.jpg")

    def detect_end_line(self):
        ret, self.frame = self.cap.read()
        if ret == False:
            return 0
        # frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.mask = detect_endline_color(self.frame)
        # if not self.magenta2m:
        #     self.magenta2m = detect_lane(self.mask,200,300)

        # else:
        #     self.magenta0m = detect_lane(self.mask,100,200)
        if not self.magenta2m:
            self.lines2 = detect_lane(self.mask,200,300)
            try:
                if len(self.lines2) > 0:
                    self.magenta2m = True
            except:
                pass

        else:
            self.lines0 = detect_lane(self.mask,400,500)
            try:
                if len(self.lines0) > 0:
                    self.magenta0m = True
            except:
                pass        

import math
def detect_lane(mask,bottom,top):
    mask = mask[bottom:top,0:mask.shape[1]]
    cv2.imshow('mask2',mask)
    edges = cv2.Canny(mask, 80, 120)
    lines = cv2.HoughLinesP(edges, 1, math.pi/2, 2, None, 30, 1)
    return lines

def detect_lane2(mask,bottom,top):

    points = []
    # print (mask.shape)
    for j in range(0,mask.shape[1],50):
        patch = mask[bottom:top,j:j+100]
        cv2.imshow('patch',patch)
        a1,a2 = patch.nonzero()
        print ("a1",a1)
        print ("a2",a2)
        if len(a1) == 0:
            continue

        count = np.count_nonzero(patch)
        if count > 500:
            points.append(j)
        # if count < 500:
        #     continue
        # pr, pv = pho(a1,a2)
        # pr = abs(pr)
        # print (pr)
        # if pr > 0.7:
        #     points.append(j)
        # else:
        #     continue   
    return len(points) > 5

def detect_endline_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # magenta
    # lower = np.array([135,60,60], dtype=np.uint8)
    # upper = np.array([165, 255, 255], dtype=np.uint8)
    # mask = cv2.inRange(hsv, lower, upper)

    # lower mask (0-10)
    lower_red = np.array([0,70,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,70,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    mask = mask0+mask1    
    return cv2.medianBlur(mask,3)


def detect_magenta(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower = np.array([135,60,60], dtype=np.uint8)
    upper = np.array([165, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def detect_red(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # lower mask (0-10)
    lower_red = np.array([0,70,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,70,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    mask = mask0+mask1
    rate = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
    return rate
if __name__ == "__main__":

    v = vision()

    while True:
        # v.start_engine()
        v.detect_end_line()

        frame = v.frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        print ("magenta2m",v.magenta2m)
        print ("magenta0m",v.magenta0m)
        if v.magenta2m:
            cv2.putText(frame,'2meters',(100, 100), font, 2,(255,0,0),2,cv2.LINE_AA)
        if v.magenta0m:
            cv2.putText(frame,'0meters',(100, 100), font, 2,(255,0,0),2,cv2.LINE_AA)
        # print (v.go)
        # if not v.green_box == []:
        #     box = v.green_box
        #     cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 3)
        #     cv2.putText(frame,'GO!',(box[0], box[1]), font, 2,(255,0,0),2,cv2.LINE_AA)
        # if not v.red_box == []:
        #     box = v.red_box
        #     cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 3)
        #     cv2.putText(frame,'STOP',(box[0], box[1]), font, 2,(255,0,0),2,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        cv2.imshow('mask',v.mask)
        k = cv2.waitKey(33)












