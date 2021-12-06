import numpy as np  # linear algebra
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import os
import cv2
import random
import imutils
import pickle
from imutils.object_detection import non_max_suppression


def illum(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def flipper(image):
    image = cv2.flip(image, 1)
    return image


# Calculate IOU
def get_iou(bb1, bb2):

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[2], bb2[2])
    x_right = min(bb1[1], bb2[1])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[1] - bb1[0]) * (bb1[3] - bb1[2])
    bb2_area = (bb2[1] - bb2[0]) * (bb2[3] - bb2[2])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


path = "/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Assignment 3/Data/PennFudanPed/PNGImages"
image_list = os.listdir(path)
image_list.sort()

# Training dataset
f = open("/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Assignment 3/pedestrian_detection/PennFudanPed_train.json")
train_dataset = json.load(f)
f.close()

train_image_list = []
train_image_to_id = {}
train_image_id_to_bbox = {}

for img in train_dataset['images']:
    img_name = img['file_name'].split('/')
    img_id = img['id']
    img_name_final = img_name[len(img_name) - 1]
    train_image_list.append(img_name_final)
    train_image_to_id[img_name_final] = img_id

for box in train_dataset['annotations']:
    bbox = box['bbox']
    image_id = box['image_id']
    if image_id not in train_image_id_to_bbox:
        train_image_id_to_bbox[image_id] = []
        train_image_id_to_bbox[image_id].append(bbox)
    else:
        train_image_id_to_bbox[image_id].append(bbox)

# Creating datset of hog features
train_X = []
train_Y = []

# Ratios of
ratios = []

# images for testing svm
images_test_svm = []

humans = 0

for imagePath in train_image_list:
    image = cv2.imread(os.path.join(path, imagePath))

    image_id = train_image_to_id[imagePath]
    bounding_boxes = train_image_id_to_bbox[image_id]
    for box in bounding_boxes:
        humans += 1
        human = image[int(box[1]):int(box[1]+box[3]),
                      int(box[0]):int(box[0]+box[2])]
        human = cv2.resize(human, (64, 128), interpolation=cv2.INTER_CUBIC)
        # Flipping the human to do data augmentation
        human_flip = flipper(human)
        fd_human = hog(human, orientations=9, pixels_per_cell=(
               8, 8), cells_per_block=(2, 2), block_norm='L2', visualize=False)
        fd_human_flipped = hog(human_flip, orientations=9, pixels_per_cell=(
            8, 8), cells_per_block=(2, 2), block_norm='L2', visualize=False)
        train_X.append(np.array(fd_human))
        train_Y.append(1)
        train_X.append(np.array(fd_human_flipped))
        train_Y.append(1)
        # # Changing the illumination for data augmentation
        # human_illum = illum(image, 2.0)
        # fd_human_illum = hog(human_illum, orientations=9, pixels_per_cell=(
        #     8, 8), cells_per_block=(2, 2), block_norm='L2', visualize=False)
        # train_X.append(np.array(fd_human_illum))
        # train_Y.append(1)
        for _ in range(10):
             x = random.randint(0, image.shape[1] - 64)
             y = random.randint(0, image.shape[0] - 128)
             while(get_iou(box, [x, x+64, y, y+128]) > 0.1):
                 x = random.randint(0, image.shape[1] - 64)
                 y = random.randint(0, image.shape[0] - 128)
             random_non_human = image[y:y+128, x:x+64]
             fd_non_human = hog(random_non_human, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', visualize=False)
             train_X.append(np.array(fd_non_human))
             train_Y.append(0)

clf = svm.SVC(kernel='linear', probability=True, random_state= 42)
clf.fit(train_X, train_Y)

filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

# Validation dataset
f = open("/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Assignment 3/pedestrian_detection/PennFudanPed_val.json")
val_dataset = json.load(f)
f.close()

val_image_list = []
val_image_to_id = {}

for img in val_dataset['images']:
    img_name = img['file_name'].split('/')
    img_id = img['id']
    img_name_final = img_name[len(img_name) - 1]
    val_image_list.append(img_name_final)
    val_image_to_id[img_name_final] = img_id

file1 = open(
    "/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Assignment 3/Predictions/hog_svm.json", 'w')
str_write = "["


def pyramid(image, scale=1.2, minSize=(30, 30)):
    curr_scale = 1
    yield (image, curr_scale)
    while True:
        w = int(image.shape[1] / scale)
        curr_scale *= scale
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield (image, curr_scale)


def sliding_window(image, stepSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + 128, x:x + 64])


for imagePath in val_image_list:
    image = cv2.imread(os.path.join(path, imagePath))

    image_id = val_image_to_id[imagePath]
    orig = image.copy()
    rects = []

    for (image_curr, scl) in pyramid(image):
        for (x, y, window) in sliding_window(image_curr, 10):
            if window.shape[0] != 128 or window.shape[1] != 64:
                continue
            hog_val = hog(window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', visualize=False)
            pred = clf.predict_proba(hog_val.reshape(1, -1))
            if(pred[0][1] >= 0.9):
                x_min = int(x*scl)
                x_max = int((x+64)*scl)
                y_min = int(y*scl)
                y_max = int((y+128)*scl)
                rects.append((x_min, y_min, x_max, y_max))

    rects = np.array([[x, y, xm, ym] for (x, y, xm, ym) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.55)
    if(len(pick) > 0):
        for rec in pick:
            cv2.rectangle(orig, (rec[0], rec[1]),
                          (rec[2], rec[3]), (255, 0, 0), 2)
            dict_img = {
                "image_id": int(val_image_to_id[imagePath]),
                "category_id": 1,
                "bbox": [int(rec[0]), int(rec[1]), int(rec[2]), int(rec[3])],
                "score": 1.0
            }
            json_object = json.dumps(dict_img)
            str_write += json_object
            str_write += ","
        write_path = "/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Assignment 3/Predictions/HOG SVM"
        cv2.imwrite(os.path.join(write_path, imagePath), orig)

str_write = str_write[0:-1]
str_write += "]"
file1.write(str_write)
file1.close()
