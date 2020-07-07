#!/usr/bin/env python
# coding: utf-8

# # Ingredient Classification




import torch
from PIL import Image
import torchvision.models as models
from torchvision import transforms
from pathlib import Path
import numpy as np
import os
from scipy.spatial.distance import cdist
from numpy import linalg as LA
import glob
import numpy as np
import matplotlib.pyplot as plt




resnext50 = models.resnext50_32x4d(pretrained=True)
resnext50_avgpooling = torch.nn.Sequential(*list(resnext50.children())[:-1])
resnext50_avgpooling.eval()




preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])





def image_to_features(image_address):
    input_image = Image.open(image_address)
    input_tensor = preprocess(input_image)
    input_batch = torch.unsqueeze(input_tensor, 0)
    out = resnext50_avgpooling(input_batch)
    out = out.squeeze(-1)
    out = out.squeeze(-1)
    outnp = out.detach().numpy()
    return outnp





def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height,im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst





labels= np.load('labels.npz')['arr_0']
features = np.load('ref_imags.npy')
#pics_path = np.load("pics_path.npz")['arr_0']



def retrieve_images(query_feature, reference_features ,labels,pics_path, num_images:int):
    distances = cdist( query_feature, reference_features, metric='euclidean' )
    sorteddist = np.argsort(distances)
    image_indices = sorteddist[0][0:num_images]
    labellist = labels[image_indices]
    labellist = labellist.tolist()
    def most_frequent(List):
        return max(set(List), key = List.count) 
    label = most_frequent(labellist)
    conf = labellist.count(label)/len(labellist)
    if conf < 0.5:
        label = 'unable to recognize'

    #img1 = Image.open(pics_path[image_indices[0]])
    #img2 = Image.open(pics_path[image_indices[1]])
    #final_image = get_concat_h(img1,img2)
    
    #for i in range(2, len(image_indices)):
    #    img = Image.open(pics_path[image_indices[i]])        
    #   final_image=get_concat_h(final_image, img)
        
    return label#,final_image
           


