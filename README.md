# Pedestrain-Detection
The aim of this project was to create a pedestrain detector from scratch, here we reproduce the research paper - "Histograms of Oriented Gradients for Human Detection" by Navneet Dalal and Bill Triggs. From each image we extract features in the form of histogram of oriented gradients(HoG) and then use these features to train a linear SVM.

## Weakness of model
To compare this model with SOTA we use the model - Faster RCNN. The weaaknesses of the model are -
- **Fixed Window size :** There is a fixed window using which the model is trained and it predicts bounding boxes of same size. Which is taken to be 64X128 here.
- **Low IoU:** Although it does a pretty decent job on predicting humans the IoU were found to be significantly lower.
- **Unable to detect incomplete or small humans:** For the SVM classifier to work, there must be complete humans with whole features visible. To deal with multiple size humans we use a multi-scale gaussian pyramid but even that is not enough to capture tiny humans.

## Predictions
Some predictions of the model are as follows- <br />
<img src="https://user-images.githubusercontent.com/12653667/144895787-2ca92e3c-bf1c-4692-aa12-59b8c8956f0f.png" width="500" height="500"> <img src="https://user-images.githubusercontent.com/12653667/144895817-5ab630b6-a56e-444f-b20f-1cc049e5a6c5.png" width="500" height="500"> <img src="https://user-images.githubusercontent.com/12653667/144895905-6bfec0f6-15f4-4c3c-add8-a1d5d30c0b6e.png" width="500" height="500">
