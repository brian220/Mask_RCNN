"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
"""
import itertools
import math
import logging
import re
import random
import matplotlib;
import matplotlib.pyplot as plt
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import imgaug

from collections import OrderedDict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("/content/drive/My Drive/Mask_RCNN_colab/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class TinyPascalConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "tiny_pascal"
    
    # Use resnet50 backbone
    BACKBONE = "resnet50"
    
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class TinyPascalDataset(utils.Dataset):

    def load_tiny_pascal(self, dataset_dir, subset):
        ann_dir = os.path.join(dataset_dir, "pascal_train.json")
        dataSet = COCO(ann_dir)
        
         # Put the train and valid in the same folder
        image_dir = os.path.join(dataset_dir, "train_images")

        # Get the class id 
        class_ids = sorted(dataSet.getCatIds())
        
        # Get the imgs id
        image_ids = list(dataSet.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("tiny_pascal", i, dataSet.loadCats(i)[0]["name"])
        
        # Add images 1000 (0 ~ 999) for training, 349 (1000 ~ 1348) for valid
        if subset == "train":
            for i in range(1000):
                index = image_ids[i]
                self.add_image(
                    "tiny_pascal", image_id=index,
                    path=os.path.join(image_dir, dataSet.imgs[index]['file_name']),
                    width=dataSet.imgs[index]["width"],
                    height=dataSet.imgs[index]["height"],
                    annotations=dataSet.loadAnns(dataSet.getAnnIds(
                        imgIds=[index], catIds=class_ids, iscrowd=None)))
        elif subset == "valid":
            for i in range(349):
                j = i + 1000
                index = image_ids[j]
                self.add_image(
                    "tiny_pascal", image_id=index,
                    path=os.path.join(image_dir, dataSet.imgs[index]['file_name']),
                    width=dataSet.imgs[index]["width"],
                    height=dataSet.imgs[index]["height"],
                    annotations=dataSet.loadAnns(dataSet.getAnnIds(
                        imgIds=[index], catIds=class_ids, iscrowd=None)))
        
    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a tiny_pascal image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "tiny_pascal":
           return super(TinyPascalDataset, self).load_mask(image_id)
        
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "tiny_pascal.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)
                    
    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


def train(model):
    # Train the model.
    # Training dataset.
    dataset_train = TinyPascalDataset()
    dataset_train.load_tiny_pascal(ROOT_DIR, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TinyPascalDataset()
    dataset_val.load_tiny_pascal(ROOT_DIR, "valid")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Tiny Pascal.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on Tiny Pascal.")
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Logs: ", args.logs)
    
     # Configurations
    if args.command == "train":
        config = TinyPascalConfig()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    
    if args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model
    
    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

     # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
    
