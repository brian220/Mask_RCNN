# Mask_RCNN
Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow
<br />
This repo is forked from https://github.com/matterport/Mask_RCNN for finishing HW4 of the Selected Topics in Visual Ricognition using DeepLearning course.
<br />
The original owner of the repo has finished the training and evaluation part.
And I add some functions that can make it more easily trained on the dataset of the HW4.
<br />
#### FILE DESCRIPTION:
* `tiny_pascal_train_head.py`:
Load the dataset and train the only head layers of the network for 100 epoches.
* `tiny_pascal.py`:
Load the dataset and fine-tune all the layers of the network in 3 stages for totally 160 epoches.

These two files are reference to the coco.py in the folder samples/coco/ of the original repo, I have done
some changes to read the json file and devide the dataset to train(1000 imgs) and valid(imgs), and both of them are put in the folder `tiny_pascal/`
