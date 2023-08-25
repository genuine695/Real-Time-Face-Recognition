from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import numpy as np
import os

preprocess_mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    keep_all=False
) # initialises MTCNN(MultiCascade Neural Network) for face detection

resnet = InceptionResnetV1(pretrained='vggface2').eval() # initialises InceptionResnetV1 for face recognition


#__________________Paths to Directory_____________________________

dataset = datasets.ImageFolder('train_image')  # photos folder path
#dataset = datasets.ImageFolder('lfw_dirs/lfw_edited_predict5/train')
BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
#IMAGE_DIR = os.path.join(BASE_DIR, 'lfw_dirs/lfw_edited_predict5/train')
TRAIN_DIR = os.path.join(BASE_DIR, 'train_image')
DATA_PATH = os.path.join(BASE_DIR, 'saveEmbeddings')
TRAIN_EMBEDS = os.path.join(DATA_PATH, 'train_image2')



label_ids = dict() # stores name and id as key-value pairs
current_id = 0

# iterates through directories | sub-directories | image files
for root, dirs, files in sorted(os.walk(TRAIN_DIR)):
    for file in files:
        #only loads file with JPEG extension
        if file.endswith("jpg"):
            path = os.path.join(root, file)
            path_name = os.path.dirname(path)
            #stores the name of the person
            label = os.path.basename(path_name)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

#reverses key-value pairs so current
dataset.labels = {v: k for k, v in label_ids.items()}
print(dataset.labels)

# stores pictures in the PIL format
loader = DataLoader(dataset, collate_fn=lambda x: x[0])

name_array = list()  # list of names corrosponding to cropped photos
embedding_array = list()  #stores embedding after image crop from mtcnn

# images and its corresponding index value is looped through the loader
for image, index in loader:
    # returns cropped face and probability of it
    face, prob = preprocess_mtcnn(image, return_prob=True)
    if face is not None:
        # pass the face into the pre-trained model and unsqueeze is because resnet needs 4 dimensions and returns embedding which is appended onto the empty array
        convertForModel = face.unsqueeze(0)
        matrix = resnet(convertForModel)
        # detach releases memory
        embedding_array.append(matrix.detach())
        # the name is saved onto name list
        name_array.append(dataset.labels[index])



    #save data as numpy array
np.savez(TRAIN_EMBEDS, x=embedding_array, y=name_array)


    # save data as PT extension
celebFaceDB = list((embedding_array, name_array))
torch.save(celebFaceDB, 'train_image2.pt')



