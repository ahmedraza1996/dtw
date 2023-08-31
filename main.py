import numpy as np
from scipy.spatial import distance
import random
import argparse
from sklearn.cluster import AgglomerativeClustering
from glob import glob
from tqdm import tqdm
from datetime import datetime
import os
import time
import torch
import torch.nn as nn
import torchvision.models as models


from feature_extractor import extract_features

def write_to_file(file_obj, string_to_write):
    file_obj.write("\n")
    file_obj.write(string_to_write)

def extract_action_segments(ground_truth):
    segments = {}
    start = None
    current_class = None
    for i, label in enumerate(ground_truth):
        if label != current_class:
            if start is not None:
                segments[current_class].append((start, i-1))
            current_class = label
            start = i
            if current_class not in segments:
                segments[current_class] = []
    if start is not None:
        segments[current_class].append((start, len(ground_truth)-1))
    return segments

def order_labels(predictions):
    k = 0 

    final_list = []
    for i in range(len(predictions)-1):
        final_list.append( k )
        if predictions[i]!=predictions[i+1]:
            k+=1
    final_list.append(k)
    return final_list
def clustering(features,n_clusters, cluster_path, prediction_path):
    # Create your data points
    X = features

    # Define the temporal neighbor threshold
    temporal_threshold = 1
    # Calculate pairwise temporal distances
    temporal_distances = np.abs(np.arange(X.shape[0]).reshape(-1, 1) - np.arange(X.shape[0]))

    # Create the custom connectivity matrix
    connectivity = temporal_distances <= temporal_threshold

    # Perform Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=connectivity)
    clustering.fit(X)

    # Get the labels assigned to each data point
    pred = clustering.labels_
    pred = order_labels(pred)
    segments =extract_action_segments(pred)
    #print(segments)
    with open(cluster_path, 'w') as file:
        for key, value in segments.items():
            line = f"{key}: {value[0][0]}, {value[0][1]}\n"
            file.write(line)

    
    with open(prediction_path, 'w') as file:
        # Write each label on a new line
        for label in pred:
            file.write(str(label) + '\n')

def run_segmentations(video_path , num_segments , output_path , cluster_path , prediction_path):


    start_time = time.time()
    if os.path.exists(output_path) and os.path.isdir(output_path):
    # Get a list of all files in the directory
        file_list = os.listdir(output_path)

        if len(file_list) > 0:
            
            for file_name in file_list:
                file_path = os.path.join(output_path, file_name)
                os.remove(file_path)
    
    resnet = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(resnet.children())[:7], resnet.avgpool).to('cuda').eval()
    
    
    extract_features(video_path, output_path , model)
    
    
    features_paths = glob(os.path.join(output_path, '*.npy'))
    file_nums = []
    for f in features_paths: 
        file_nums.append(int(f.split('/')[-1].split('.')[0]))

    def extract_numeric_part(file_path):
        return int(file_path.split('/')[-1].split('.')[0])

    # Use glob to sort the file paths based on increasing numbers
    sorted_file_paths = sorted(features_paths, key=extract_numeric_part)


    features = []

    # Load the arrays from the file paths
    for file_path in sorted_file_paths:
        array = np.load(file_path)
        features.append(array)

    features = np.concatenate(features)
    
    
    # self.features = np.load(feature_path)
    #self.features= np.transpose( self.features)
    print("Features during loading")
    print(features.shape)


    clustering(features,num_segments,  cluster_path , prediction_path)
    end_time = time.time()
    time_Delta = end_time-start_time
    print(" FRAME RATE: ", len(features_paths)/time_Delta)



if __name__ =='__main__':
    #path to video
    base_name = '2020-12-1817-08-32.mp4'
    
    video_path = "/home/ahmed/Ahmed_data/agglomerative/harddrive/2023-01-2116-12-50.mp4"
    #path where npy files for each feature will be saved
    output_path = '/home/ahmed/Ahmed_data/Sateesh ResNet/test/output/'
    # file path where clusters and predictions will be saved ( prediction -> clusters.txt, predictions.txt)
    cluster_path = '/home/ahmed/Ahmed_data/agglomerative/clusters/clusters_22023-01-2116-12-50.mp4'
    prediction_path  = '/home/ahmed/Ahmed_data/agglomerative/predictions/predictions_2023-01-2116-12-50.mp4'
    # number of action segments
    num_segments= 23

    run_segmentations(video_path, num_segments, output_path,cluster_path , prediction_path)