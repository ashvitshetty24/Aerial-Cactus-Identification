#! /usr/bin/python

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
import glob
import random


img_dir = (r"C:\Users\shett\.spyder-py3\Final Project\train\*.jpg") 
file_names=[]
files = glob.glob(img_dir)
print(len(files))
for i in range(17500):
    path_array = files[i].split("\\")
    file_names.append(path_array[-1])

data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)

dataset = np.asarray(data)
dataset = dataset.reshape(dataset.shape[0],-1).T
print(dataset.shape)

labels = pd.read_csv("train.csv")
labels = labels.values
ground_truth = []
for image_name in file_names:
    for i in range(len(file_names)):
        if image_name == labels[i][0]:
            ground_truth.append(labels[i][1])
            break
ground_truth = np.asarray(ground_truth)
#print(ground_truth.shape)
ground_truth = ground_truth.reshape(ground_truth.shape[0], 1).T
print(ground_truth.shape)


random.seed(4)
random.shuffle(ground_truth)
random.shuffle(dataset)
random.seed(4)

count = int(0.8*dataset.shape[1])
train_features = dataset[:, :count]
train_label = ground_truth[:, :count]
test_features = dataset[:, count:]
test_label = ground_truth[:, count:]

print(train_features.shape)
print(train_label.shape)
print(test_features.shape)
print(test_label.shape)
