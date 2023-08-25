from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import cv2
import os
from PIL import Image
import time


#___________________code to take pictures and put the in folders________________________

''''
number = 1

#path to testVidDirectory where most of the videos are stored
path = "testVidDirectories/testVidDirectory" + str(number)

#iterates through the files in that directory
for file in os.listdir(path):
    #skips over .DS_STORE file
    if file.startswith('.'):
        continue
    print(file[12:].replace(".avi",""))

    # reads current path file
    cap = cv2.VideoCapture(path + '/' + file)
    count = 0

    currentFrame = 0
    while True:
        ret, frame = cap.read()
        new_path = 'data/data' + str(number)
        if ret == True:
            name = file[12:].replace(".avi", "")
            new_dir = os.path.join(new_path, file[12:].replace(".avi", ""))
            # makes directory if the database did not exist before
            if not os.path.isdir(new_dir):
                os.makedirs(new_dir)
            # takes 20 pictures
            if currentFrame < 20:
                frame_name = os.path.join(new_path,file[12:].replace(".avi",""))+'/frame' + str(currentFrame) + '.jpg'
                cv2.imwrite(frame_name, frame)

        currentFrame += 1
        if not ret:
            break

    cap.release()
    cv2.destroyAllWindows()

'''

#_______ save dataset as .pt file extension_____


select_data = 5

preprocess_mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    keep_all=False
)

resnet = InceptionResnetV1(pretrained='casia-webface').eval()

#path to directories and files

dataset = datasets.ImageFolder('data/data' + str(select_data))  # photos folder path
BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
IMAGE_DIR = os.path.join(BASE_DIR, 'data/data' + str(select_data))

label_ids = dict() # stores name and id as key-value pairs
current_id = 0

# iterates through directories | sub-directories | image files
for root, dirs, files in sorted(os.walk(IMAGE_DIR)):
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root, file)
            path_name = os.path.dirname(path)
            label = os.path.basename(path_name)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

#reverses key-value pairs so current
dataset.labels = {v: k for k, v in label_ids.items()}
print(dataset.labels)


# stores pictures in the PIL format
loader = DataLoader(dataset, collate_fn=lambda x: x[0])

name_array = []  # list of names corrosponding to cropped photos
embedding_array = []  # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

# images and its corresponding index value is looped through the loader
for image, index in loader:
    # returns cropped face and probability of it
    face, prob = preprocess_mtcnn(image, return_prob=True)
    if face is not None:
        # pass the face into the pre-trained model and unsqueeze is because resnet needs 4 dimensions and returns embedding which is appended onto the empty array
        convertForModel = face.unsqueeze(0)
        emb = resnet(convertForModel)
        # detach releases memory
        embedding_array.append(emb.detach())
        # the name is saved onto name list
        name_array.append(dataset.labels[index])

    # save data read from folder
celebFaceDB = list((embedding_array, name_array))
torch.save(celebFaceDB, 'save_video_model/video_dataset_casia{number}.pt'.format(number=select_data))



''''
#_____________________code to arrange ytcelebrity dataset_____________________


dir = 'ytcelebrity'
for x in os.listdir(dir):
    if x.startswith('.'):
        continue
    else:
        file_name = x[12:]
        name = file_name.replace(".avi", "")
        path = os.path.join(dir,name)
        if not os.path.isdir(path):
            path = os.path.join(dir,name)
            os.mkdir(path)

for file in os.listdir(dir):
    path = os.path.join(dir,file)
    if file.startswith('.') or not os.path.isfile(path):
        continue
    else:
        full_file_name = file
        file_name = file[12:]
        name = file_name.replace(".avi", "")
        ex_path = os.path.join(dir, file)
        new_path = os.path.join(dir, name, file)
        os.replace(ex_path, new_path)

'''