from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import json
import os
from os import listdir
from os.path import isfile, join

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

path = "/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Assignment 3/Data/PennFudanPed/PNGImages"
image_list = os.listdir(path)
image_list.sort()

f = open("/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Assignment 3/pedestrian_detection/PennFudanPed_val.json")
val_dataset = json.load(f)
f.close()

image_list = []
image_to_id = {}

for img in val_dataset['images']:
	img_name = img['file_name'].split('/')
	img_id = img['id']
	img_name_final = img_name[len(img_name)- 1]
	image_list.append(img_name_final)
	image_to_id[img_name_final] = img_id


file1 = open("/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Assignment 3/Predictions/hog_original.json",'w') 
str_write = "["

for imagePath in image_list:
	if imagePath not in image_list:
		continue
	image = cv2.imread(os.path.join(path,imagePath))
	orig = image.copy()
	
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
	
	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
	
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
	if imagePath in image_list:
		dict_img = {
			"image_id" : int(image_to_id[imagePath]),
			"category_id" : 1,
			"bbox" : [float(x),float(y),float(w),float(h)],
			"score" : 1.0
		}
		json_object = json.dumps(dict_img)
		str_write += json_object
		str_write += ","
	

	write_path = "/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Assignment 3/Predictions/HOG Original"
	cv2.imwrite(os.path.join(write_path,imagePath), image)

str_write = str_write[0:-1]
str_write += "]"
file1.write(str_write)
file1.close()