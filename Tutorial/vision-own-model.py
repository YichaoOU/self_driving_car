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



def detect_traffic_lights(image_cv2,sess):

    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_np_expanded = np.expand_dims(image_cv2, axis=0)
    print (image_np_expanded.shape)
    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
      [detection_boxes, detection_scores, detection_classes, num_detections],
      feed_dict={image_tensor: image_np_expanded})
    go_flag = False
    if classes[0][0] == 2.0:
        go_flag = True   
    im_width=image_cv2.shape[1]
    im_height = image_cv2.shape[0]
    ymin, xmin, ymax, xmax = boxes[0][0].tolist()
    left, right, top, bottom = map(lambda x:int(x),[xmin * im_width, xmax * im_width,ymin * im_height, ymax * im_height])

    return go_flag,[left, right, top, bottom]

class vision():
    def __init__(self):
        # change it to the camera for real application
        # self.cap = cv2.VideoCapture("IGVC_2015_Speed_Record.mp4")
        self.start_time = time.time()
        self.cap = cv2.VideoCapture("stocker1f-test1.mp4")
        self.time_stamp = 0
        self.go = False
        self.magenta2m = False
        self.magenta0m = False




    def start_engine(self,sess):
        # for traffic light detection
        ret,self.frame = self.cap.read()
        # self.frame = cv2.resize(self.frame,(300,300))
        if ret == False:
            print ("Nothing read in")
            return 0
        # frame = cv2.imread("test_images/test3.jpg")
        self.go,self.boxes_list = detect_traffic_lights(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB),sess)
        # self.go = detect_traffic_lights("test_images/test3.jpg")

    def detect_end_line(self):
        ret, frame = self.cap.read()
        if ret == False:
            return 0
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = detect_magenta_color(frame)
        if not self.magenta2m:
            self.magenta2m = detect_lane(mask,200,300)
        else:
            self.magenta0m = detect_lane(mask,100,200)

def detect_lane(mask,bottom,top):

    points = []
    for j in range(0,1200,50):
        patch = mask[bottom:top,j:j+100]
        a1,a2 = patch.nonzero()
        count = np.count_nonzero(patch)
        if count < 500:
            continue
        pr, pv = pho(a1,a2)
        pr = abs(pr)
        
        if pr > 0.7:
            points.append(j)
        else:
            continue   
    return len(points) > 5

def detect_magenta_color(frame):
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    r1 = np.array([150,0,130], dtype=np.uint8)
    r2 = np.array([255, 100, 255], dtype=np.uint8)
    mask = cv2.inRange(frame, r1, r2)
    return cv2.medianBlur(mask,3)
# need 4 seconds to process a frame in my PC
# it doesn't depends on the input size
if __name__ == "__main__":

    v = vision()

    """
    Detect traffic lights and draw bounding boxes around the traffic lights
    :param PATH_TO_TEST_IMAGES_DIR: testing image directory
    :param MODEL_NAME: name of the model used in the task
    :return: commands: True: go, False: stop
    """

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = 'traffic_inference_graph/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'traffic_inference_graph/traffic_light.pbtxt'

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

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            while True:
                v.start_engine(sess)
                frame = v.frame.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                box = v.boxes_list
                print (box)
                print (v.go)
                # [left, right, top, bottom]
                # cv2.rectangle(frame, (box[2], box[0]), (box[3], box[1]), (0, 128, 255), 1)
                cv2.rectangle(frame, (box[0], box[2]), (box[1], box[3]), (0, 128, 255), 2)
                if v.go:
                    cv2.putText(frame,'GO!',(box[0], box[2]), font, 2,(255,0,0),2,cv2.LINE_AA)
                else:
                    cv2.putText(frame,'STOP',(box[0], box[2]), font, 2,(255,0,0),2,cv2.LINE_AA)
                cv2.imshow('frame',frame)
                k = cv2.waitKey(33)












