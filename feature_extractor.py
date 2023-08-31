import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os, glob
import numpy as np
import cv2
import time

def extract_features(video_path, output_path,   model =None ):
    
   
    if not os.path.exists(output_path): 
        os.mkdir(output_path) 
    torch.set_grad_enabled(False)
    

    transform  = transforms.Compose([transforms.Resize((256,480)),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

    cap = cv2.VideoCapture(video_path)
    

    #start_time = time.time()
    count= 0 
    # Extract frame-wise features
    frame_features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB format
        image_pil = Image.fromarray(frame)
        frame = transform(image_pil)
        frame = torch.unsqueeze(frame, dim=0)  # Add batch dimension
        frame= frame.to(device= 'cuda')
        features = model(frame)
        features = torch.squeeze(features).unsqueeze(0)
        #print(" features shape: ", features.shape)
        filename ='{:05d}.npy'.format(count)
        np.save(os.path.join(output_path , filename) , features.detach().cpu().numpy() )
        #frame_features.append(frame)
        count+=1

    #end_time = time.time()
    
    # Release video capture and close file
    cap.release()
    #print(" FRAME RATE: ", count/(end_time-start_time))

   

def generate_FRAMES(video_path):
    cap = cv2.VideoCapture(video_path)
    dir_path = video_path.split(".")[0]
    os.mkdir(dir_path)
    counter = 0
    while True:
        ret, frame = cap.read()

        if ret:
            cv2.imwrite(dir_path + "/%06d.png"%(counter), frame)
            counter +=1
        else:
            break

def load_Features(src, dest):
    for feature_path in os.listdir(src):

        feature = np.load(src + "/" + feature_path)
        print(feature.shape)
       

if __name__=='__main__': 
    video_path = '/home/ahmed/Ahmed_data/Murad_branches/fast_and_unsupervised_action_boundary_detection_for_action_segmentation/data/apple/videos/01_fast.mp4'
    
    output_path = '/home/ahmed/Ahmed_data/Murad_branches/fast_and_unsupervised_action_boundary_detection_for_action_segmentation/data/output'
    
    
    if os.path.exists(output_path) and os.path.isdir(output_path):
    # Get a list of all files in the directory
        file_list = os.listdir(output_path)

        if len(file_list) > 0:
           
            for file_name in file_list:
                file_path = os.path.join(output_path, file_name)
                os.remove(file_path)
            
            
    resnet = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(resnet.children())[:7]).to('cuda').eval()
    extract_features(video_path, output_path , model)
    