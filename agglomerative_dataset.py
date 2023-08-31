import numpy as np
from scipy.spatial import distance
import seaborn as sns
import random
from sklearn_extra.cluster import KMedoids
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import random
from evaluation.accuracy import Accuracy
from evaluation.f1_score import F1Score
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import os

def bounds(segm):
    start_label = segm[0]
    
    start_idx = 0
    idx = 0
    while idx < len(segm):
        try:
            while start_label == segm[idx]:
                idx += 1
        except IndexError:
            yield start_idx, idx, start_label
            break

        yield start_idx, idx, start_label
        start_idx = idx
        start_label = segm[start_idx]


def load_ground_truth(file_path):
    with open(file_path, 'r') as file:
        ground_truth = [line.strip() for line in file.readlines()]
    return ground_truth
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

def sample_random_index(segment):
    start, end = segment
    return random.randint(start, end)

def sample_random_indices(action_segments):
    sampled_indices = {}
    for action_class, segments in action_segments.items():
        sampled_indices[action_class] = [sample_random_index(segment) for segment in segments]
    return sampled_indices
def pick_middle_index(segment):
    start, end = segment
    return (start + end) // 2

def pick_middle_indices(action_segments):
    middle_indices = {}
    for action_class, segments in action_segments.items():
        middle_indices[action_class] = [pick_middle_index(segment) for segment in segments]
    return middle_indices







if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg', default=False, action='store_true', help='for YTI, if dataset has background')
    args = parser.parse_args()
    parent_dir  = '/home/ahmed/Ahmed_data/DATASETS/data/DesktopAssembly/'
    gt_dir = parent_dir+'groundTruth/'
    features_dir = parent_dir+'features_resnet50_1024/'
    files = os.listdir(gt_dir)
    
    mofs = []
    f1s  = []

    for f in files: 
        feat_path = features_dir+f+'.npy'
        gt_path = gt_dir + f
    # Example usage
   
        ground_truth = load_ground_truth(gt_path)
        feat = np.load(feat_path)
        
        K=23

        mapping_file = "/home/ahmed/Ahmed_data/DATASETS/data/DesktopAssembly/mapping/mapping.txt"
        # reading and mapping the names (action names) and classes (action ids)
        file_ptr = open(mapping_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        actions_dict = dict()
        # actions_dict
        for a in actions:
            actions_dict[a.split()[1]] = int(a.split()[0])

        classes = np.zeros(min(feat.shape[0], len(ground_truth)),dtype=np.float32)
        for i in range(len(classes)):
            classes[i] = actions_dict[ground_truth[i]] #content[i]

        frames = min(feat.shape[0] , len(classes))
        feat= feat[:frames]
        classes = classes[:frames]
        

        # Create your data points
        X = feat

        # Define the temporal neighbor threshold
        temporal_threshold =1
        # Calculate pairwise temporal distances
        temporal_distances = np.abs(np.arange(X.shape[0]).reshape(-1, 1) - np.arange(X.shape[0]))

        # Create the custom connectivity matrix
        connectivity = temporal_distances <= temporal_threshold

        # Perform Agglomerative Clustering
        clustering = AgglomerativeClustering(n_clusters=K, linkage='ward', connectivity=connectivity)
        clustering.fit(X)

        # Get the labels assigned to each data point
        pred = clustering.labels_

        
        


        accuracy  = Accuracy(args)
        accuracy.predicted_labels = pred
        accuracy.gt_labels = classes
        old_mof, total_fr = accuracy.mof()

        label2gt ={}
        gt2label = accuracy._gt2cluster
        for key, val in gt2label.items():
            try:
                label2gt[val[0]] = key
            except IndexError:
                pass
        print("label to gt: " , label2gt)
        print('MoF val: ' + str(accuracy.mof_val())) # mof without bg
        print('old MoF val: ' + str(float(old_mof) / total_fr))

        mofs.append(accuracy.mof_val())
        # Equivalent to Accuracy corpus after generating prototype likelihood
        f1_score = F1Score(K=K, n_videos=1)
        gt2label = accuracy._gt2cluster
        accuracy  = Accuracy(args)
        accuracy.gt_labels = classes
        accuracy.predicted_labels = pred
        f1_score.set_gt(classes)
        f1_score.set_pr(pred)
        f1_score.set_gt2pr(gt2label)

        
        print(" second") 
        print(len(accuracy.exclude))  
        old_mof, total_fr = accuracy.mof(old_gt2label=gt2label)
        gt2label = accuracy._gt2cluster

        
        label2gt = {}
        for key, val in gt2label.items():
            try:
                label2gt[val[0]] = key
            except IndexError:
                pass
        acc_cur = accuracy.mof_val()
        print(f"MoF val: {acc_cur}")
        print(f"previous dic -> MoF val: {float(old_mof) / total_fr}")
        average_class_mof = accuracy.mof_classes()
        return_stat = accuracy.stat()       
        print("MOF STATS: " , return_stat)
        
        f1_score.f1()
        for key, val in f1_score.stat().items():
            print("key :  " , key , "val : " , val)
            return_stat[key] = val
            if key=='mean_f1':
                f1s.append(val[0])

    print(np.mean(mofs))
    print(np.mean(f1s))
    print("f1s: ", f1s)
    print(" mofs:  ", mofs)
    