import numpy as np
from scipy.spatial import distance
import seaborn as sns
import random
from sklearn_extra.cluster import KMedoids
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import random

import argparse
import matplotlib.pyplot as plt

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

def order_labels(predictions):
    k = 0 

    final_list = []
    for i in range(len(predictions)-1):
        final_list.append( k )
        if predictions[i]!=predictions[i+1]:
            k+=1
    final_list.append(k)
    return final_list






if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg', default=False, action='store_true', help='for YTI, if dataset has background')
    args = parser.parse_args()
    base_name = 'clamp-lamp-assembly__2020-12-1714-52-10.mp4'
    feat = np.load(f'/home/ahmed/Ahmed_data/agglomerative/features/{base_name}.npy')
    cluster_path = f'/home/ahmed/Ahmed_data/agglomerative/clusters/clusters_{base_name}'
    prediction_path  = f'/home/ahmed/Ahmed_data/agglomerative/predictions/predictions_{base_name}'

    # Example usage
    #file_path = '/home/ahmed/Ahmed_data/DATASETS/data/DesktopAssembly/groundTruth/2020-04-02-150120'
    #ground_truth = load_ground_truth(file_path)

    timestamp_file_path = '/home/ahmed/Ahmed_data/agglomerative/predictions/predictions'
    timestamp_ground_truth = load_ground_truth(timestamp_file_path)

    
    action_segments = extract_action_segments(timestamp_ground_truth)
    print(action_segments)
    K=17
    # mapping_file = "/home/ahmed/Ahmed_data/DATASETS/data/DesktopAssembly/mapping/mapping.txt"
    #     # reading and mapping the names (action names) and classes (action ids)
    # file_ptr = open(mapping_file, 'r')
    # actions = file_ptr.read().split('\n')[:-1]
    # file_ptr.close()
    # actions_dict = dict()
    # # actions_dict
    # for a in actions:
    #     actions_dict[a.split()[1]] = int(a.split()[0])

    # classes = np.zeros(min(feat.shape[0], len(ground_truth)),dtype=np.float32)
    # for i in range(len(classes)):
    #     classes[i] = actions_dict[ground_truth[i]] #content[i]

    random_indices = pick_middle_indices(action_segments)
    print(random_indices)



    timestamps = []
    for k, v in random_indices.items(): 
        timestamps.append(v[0])

    medoids = []
    for t in timestamps: 
        medoids.append(feat[t])
        
    medoids = np.array(medoids)
    print(timestamps)
    counter = 0 
    while counter <1:
        boundaries = []
        boundaries.append(0)
        distances = distance.cdist(medoids, feat)
        #print(distances.shape)
        for i in range(distances.shape[0]-1):
            distance_list = []
            for l in range(timestamps[i],timestamps[i+1]):
                distance_list.append(np.sum(distances[i, timestamps[i]:l+1])+np.sum(distances[i+1,l+1:timestamps[i+1]]))
            idx = np.argmin(distance_list)

            boundaries.append(timestamps[i]+idx) 
        boundaries.append(distances.shape[1])
        timestamps =[]
        new_medoids = []
        for i in range(distances.shape[0]):
            intra_distances= distance.cdist(feat[boundaries[i]:boundaries[i+1]], feat[boundaries[i]:boundaries[i+1]])
            total_distances = np.sum(intra_distances, axis=1)
            new_medoid_index = np.argmin(total_distances)+boundaries[i]
            timestamps.append(new_medoid_index)

            new_medoid = feat[new_medoid_index]
            new_medoids.append(new_medoid)

        counter+=1
        if np.array_equal(medoids, np.array(new_medoids)):
            break
        else:
            print(timestamps)
            medoids = new_medoids

    
    pred = []
    value =  0 
    for i in range(len(boundaries)-1):
        end = boundaries[i+1]
        start= boundaries[i]
        iters = end -start
        for j in range(iters): 
            pred.append(value)
        value +=1
    
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

    colors = {}
    cmap = plt.get_cmap('tab20')
    for label_idx, label in enumerate(np.unique(pred)):
        if label == -1:

            colors[label] = (0, 0, 0)
        else:
            # colors[label] = (np.random.rand(), np.random.rand(), np.random.rand())
            colors[label] = cmap(label_idx / len(np.unique(pred)))
            
    
