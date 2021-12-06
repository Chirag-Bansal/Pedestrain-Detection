import warnings
warnings.filterwarnings("ignore")
import torchvision
from torchvision.models import detection
import numpy as np
import argparse
import pickle
import json
import torch
import cv2
import os
from matplotlib import pyplot as plt
import sys
from torch.utils.data import Dataset, DataLoader

# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PedestrianDataset(Dataset):
	"""Creating a datset of images from path of images"""

	def __init__(self, root_dir, image_list):
		self.root_dir = root_dir
		self.image_list = image_list

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self,index):
		img_path = os.path.join(self.root_dir,self.image_list[index])
		orig = image.copy()
		image = cv2.imread(img_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.transpose((2, 0, 1))
		image = np.expand_dims(image, axis=0)
		image = image / 255.0
		image = torch.FloatTensor(image)
		image = image.to(DEVICE)
		return (orig,image)


path = "/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Assignment 3/Data/PennFudanPed/PNGImages"
image_list = os.listdir(path)

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

file1 = open("/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Assignment 3/Predictions/faster_rcnn.json",'w')
str_write = "["


model = detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True,
	num_classes=91, pretrained_backbone=True).to(DEVICE)

model.eval()

for imagePath in image_list:
	total_path = os.path.join(path,imagePath)
	image = cv2.imread(total_path)
	orig = image.copy()

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.transpose((2, 0, 1))
	image = np.expand_dims(image, axis=0)
	image = image / 255.0
	image = torch.FloatTensor(image)
	image = image.to(DEVICE)
	detections = model(image)[0]

	# loop over the detections
	for i in range(0, len(detections["boxes"])):
		confidence = detections["scores"][i]
		if confidence > 0.99:
			idx = int(detections["labels"][i])
			if(idx != 1):
				continue
			box = detections["boxes"][i].detach().cpu().numpy()
			(startX, startY, endX, endY) = box.astype("int")

			cv2.rectangle(orig, (startX, startY), (endX, endY),	(0,0,255), 2)

			if imagePath in image_list:
				dict_img = "{" + "\"image_id\"" + ":" +	str(image_to_id[imagePath]) + "," + "\"category_id\"" + ":" + str(1) + "," + "\"bbox\"" + ":" + "[" + str(float(startX)) +"," + str(float(startY)) + "," + str(float(endX-startX)) + "," + str(float(endY-startY)) +"]" + "," + "\"score\"" + ":" + str(1.0) +"}"
				str_write += dict_img
				str_write += ","

	write_path = "/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Assignment 3/Predictions/RCNN"
	cv2.imwrite(os.path.join(write_path,imagePath), orig)

str_write = str_write[0:-1]
str_write += "]"
file1.write(str_write)
file1.close()
