import cv2
import sys
import os
import shutil
file = sys.argv[1]
label = file.replace(".mp4","")
vidcap = cv2.VideoCapture(file)
success,image = vidcap.read()
count = 0
# os.mkdir(label+"_frames")
while success:
  count += 1
  if count % 10 == 0:
    cv2.imwrite(label+"%d.jpg" % count, image)
    shutil.move(label+"%d.jpg" % count, "training_images")
  success,image = vidcap.read()
  