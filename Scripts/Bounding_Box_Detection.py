#!/usr/bin/env python
# coding: utf-8

# # Bounding Box Detection and Ingredient Classification

# In[1]:


from models import *
from utils import *
from Ingredient_Classification import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import streamlit as st


# In[2]:


config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.001
nms_thres=0.3

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
#model.cuda()
model.eval()
classes = utils.load_classes(class_path)
#Tensor = torch.cuda.FloatTensor
Tensor = torch.FloatTensor


# resizing the image to a 416px square while maintaining its aspect ratio and padding the overflow.

# In[3]:


def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


# In[16]:

#@st.cache
def ingredient_detection(img_path):
        # load image and get detections
        prev_time = time.time()
        img = Image.open(img_path)
        ingredients = [] 
        detections = detect_image(img)
        inference_time = datetime.timedelta(seconds=time.time() - prev_time)
        #print ('Inference Time: %s' % (inference_time))

        # Get bounding-box colors
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        img = np.array(img)
        plt.figure()
        fig, ax = plt.subplots(1, figsize=(12,9))
        ax.imshow(img)

        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            # browse detections and draw bounding boxes
            d = 0
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections: 

                b_h = y2 - y1 
                b_w = x2 - x1 


                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]

                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                x2 = x1 + box_w
                y2 = y1 + box_h

                if b_h*b_w>=3000 and (b_h<250 and b_w <250) :
                    d += 1
                    top_left_x = max(int(min([x1,x2]).item()),1)            
                    top_left_y = max(int(min([y1,y2]).item()),0)
                    bot_right_x = min(int(max([x1,x2]).item()),img.shape[1]-1)
                    bot_right_y = min(int(max([y1,y2]).item()),img.shape[0])

                    cropped_image = img[top_left_y:bot_right_y, top_left_x:bot_right_x]
                    #plt.figure()
                    #fig, ax = plt.subplots(1, figsize=(12,9))
                    #ax.imshow(cropped_image)
                    name = '_'+str(d)+'.jpg'
                    file_name = 'selected_pic.jpg'.replace(".jpg", name)
                    cv2.imwrite(file_name,cropped_image[:,:,::-1])
                    img_feature = image_to_features(file_name)
                    cl = retrieve_images(img_feature,features,labels, pics_path, 11)
                    #print(cl)
                    #fig, ax = plt.subplots(1, figsize=(12,9))
                    #ax.imshow(similar_images)
                    if cl!= 'unable to recognize':
                        ingredients.append(cl)



                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(bbox)
                    plt.text(x1, y1, s=cl, color='white', verticalalignment='top', fontsize=16,
                            bbox={'color': color, 'pad': 0})
        plt.axis('off')
        # save image
        file_name = 'selected_pic.jpg'.replace(".jpg", "det1.jpg")
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0.0)
        #plt.show()
        
        return file_name,list(set(ingredients))

