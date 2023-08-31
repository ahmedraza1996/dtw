import json
import pickle
import statistics

import numpy as np
import shutil

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

from deep_models.dl_models.video_segmentation.feature_extractor import extract_features


def clean_output_dir(output_path):
	if os.path.exists(output_path):
		for file in os.listdir(output_path):
			if file.endswith(".npy"):
				os.remove(os.path.join(output_path, file))


def extract_action_segments(ground_truth):
	segments = {}
	start = None
	current_class = None
	for i, label in enumerate(ground_truth):
		if label != current_class:
			if start is not None:
				segments[current_class] = [start, i - 1]
			current_class = label
			start = i
			if current_class not in segments:
				segments[current_class] = []
	if start is not None:
		segments[current_class] = [start, len(ground_truth) - 1]
	return segments


def order_labels(predictions):
	k = 0

	final_list = []
	for i in range(len(predictions) - 1):
		final_list.append(k)
		if predictions[i] != predictions[i + 1]:
			k += 1
	final_list.append(k)
	return final_list


def clustering(features, n_clusters, cluster_path):
	if not os.path.exists(cluster_path):
		os.makedirs(cluster_path,exist_ok=True)
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
	segments = extract_action_segments(pred)
	# print(segments)
	with open(os.path.join(cluster_path,"cluster.json"), 'w') as file:
		json.dump(segments,file)

	return segments

		# for key, value in segments.items():
		# 	line = f"{key}: {value[0][0]}, {value[0][1]}\n"
		# 	file.write(line)

	# with open(prediction_path, 'w') as file:
	# 	# Write each label on a new line
	# 	for label in pred:
	# 		file.write(str(label) + '\n')


def run_segmentations(video_path, num_segments, intermediate_path, cluster_path, progress_marker):
	output_path = intermediate_path
	clean_output_dir(output_path)
	if not os.path.exists(output_path):
		os.makedirs(output_path,exist_ok=True)
	if progress_marker:
		progress_marker.progress = 2
		progress_marker.save()
	resnet = models.resnet50(pretrained=True)
	model = nn.Sequential(*list(resnet.children())[:7], resnet.avgpool).to('cuda').eval()
	extract_features(video_path, output_path, model,progress_marker)
	features_paths = glob(os.path.join(output_path, '*.npy'))
	file_nums = []
	for f in features_paths:
		file_nums.append(os.path.basename(f))

	def extract_numeric_part(file_path):
		return int(os.path.basename(file_path).split('.')[0])

	# Use glob to sort the file paths based on increasing numbers
	sorted_file_paths = sorted(features_paths, key=extract_numeric_part)

	features = []

	# Load the arrays from the file paths
	for file_path in sorted_file_paths:
		array = np.load(file_path)
		features.append(array)
	conc_features = np.concatenate(features)
	segments = clustering(conc_features, num_segments, cluster_path)
	# upper_limit = segments[list(segments.keys())[-1]][1]
	# print(segments)
	# segments = {}
	# index = 0
	# frames = 60
	# for i in range(frames, upper_limit, frames):
	# 	segments[index] = [i - frames, i]
	# 	index += 1
	# if i < upper_limit:
	# 	segments[index] = [i, upper_limit]
	# print(segments)
	segment_path = os.path.join(cluster_path, "segments")
	os.makedirs(segment_path, exist_ok=True)
	create_segment_videos(video_path, segment_path, segments)
	# average_features = {}
	features = np.reshape(features,(len(features),1024))
	# for name,segment in segments.items():
	# 	average_features[name] = list(np.mean(features[segment[0]:segment[1]+1],axis=0))
	compare_features_with_base_video(features,segments,segment_path)
	clean_output_dir(output_path)
	if progress_marker:
		progress_marker.progress = 100
		progress_marker.save()

def compare_features_with_base_video(base_features,segments,segment_path):
	import shutil
	segments_dir = r"C:\Users\NomanAnjum\Downloads\Segmentation Test\video1\segments"
	target_dir = r"C:\Users\NomanAnjum\Downloads\Segmentation Test\Smiliar Segments"
	for file in os.listdir(segments_dir):
		folder_name = file.split(".")[0]
		os.makedirs(os.path.join(target_dir,folder_name),exist_ok=True)
		shutil.copy(os.path.join(segments_dir,file),os.path.join(target_dir,folder_name,file))

	with open(r"C:\Users\NomanAnjum\Downloads\Segmentation Test\features","rb") as f:
		target_features = pickle.load(f)
	print(target_features)
	for base_key,segment in segments.items():
		nearest_neighbours = []
		for i in range(segment[0],segment[1]+1):
			base_feature = base_features[i]
			smallest_distance = None
			nearest_segment = None
			for target_key,target_feature in target_features.items():

				distances = np.linalg.norm(np.array(target_feature) - np.array(base_feature), axis=1)
				sorted_indices = np.argsort(distances)
				if smallest_distance == None:
					smallest_distance = distances[sorted_indices[0]]
					nearest_segment = target_key
				else:
					if distances[sorted_indices[0]] < smallest_distance:
						smallest_distance = distances[sorted_indices[0]]
						nearest_segment = target_key
			nearest_neighbours.append(nearest_segment)
		nearest_neighbour = statistics.mode(nearest_neighbours)
		file = os.path.join(segment_path,f"segment_{base_key}.mp4")
		shutil.copy(file,os.path.join(target_dir,f"segment_{nearest_neighbour}",f"video2_{base_key}.mp4"))
	# 	for name,features in base_features.items():
	# 		distances = np.linalg.norm(np.array(list(target_features.values())) - np.array(features), axis=1)
	# 		sorted_indices = np.argsort(distances)
	# 		print(f"Name , {name} \n",sorted_indices)
	# 		file = os.path.join(segment_path,f"segment_{name}.mp4")
	# 		shutil.copy(file,os.path.join(target_dir,f"segment_{sorted_indices[0]}",f"video2_{name}.mp4"))

def create_segment_videos(video_path,segment_path,segments):
	import cv2
	cap = cv2.VideoCapture(video_path)
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	video_fps = int(cap.get(cv2.CAP_PROP_FPS))
	# start_time = time.time()
	count = 0
	# Extract frame-wise features
	frame_features = []
	curr_seg_num = 0
	size = (640, 480)
	curr_segment = list(segments.values())[curr_seg_num]
	result = cv2.VideoWriter(f'{segment_path}/segment_{curr_seg_num}.mp4',
							 cv2.VideoWriter_fourcc(*'mp4v'),
							 video_fps, size)
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		if count <= curr_segment[1]:
			result.write(frame)
		else:
			print(f"Video {curr_seg_num} complete")
			curr_seg_num+=1
			curr_segment = list(segments.values())[curr_seg_num]
			result = cv2.VideoWriter(f'{segment_path}/segment_{curr_seg_num}.mp4',
							 cv2.VideoWriter_fourcc(*'mp4v'),
							 video_fps, size)
			result.write(frame)
		count+=1
	result.release()
	cap.release()

if __name__ == '__main__':
	# path to video
	video_path = r"C:\Users\NomanAnjum\Downloads\Segmentation Test\video2\w1_c1.mp4"
	# path where npy files for each feature will be saved
	output_path = r"C:\Users\NomanAnjum\Downloads\Segmentation Test\video2\intermediate"
	# file path where clusters and predictions will be saved ( prediction -> clusters.txt, predictions.txt)
	cluster_path = r"C:\Users\NomanAnjum\Downloads\Segmentation Test\video2"
	prediction_path = "D:\Ahmed Data\output"
	# number of action segments
	num_segments = 22

	run_segmentations(video_path, num_segments, output_path, cluster_path, None)
