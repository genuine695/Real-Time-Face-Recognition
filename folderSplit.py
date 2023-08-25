import splitfolders
import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import cv2
import time
import numpy as np
import os
import glob

#splits folder into train/test sets
'''
splitfolders.ratio("lfw_edited", output="lfw_dirs/lfw_edited _predic1t", seed=12, ratio=(.8, .2))
splitfolders.ratio("lfw_edited", output="lfw_dirs/lfw_edited _predict2", seed=1337, ratio=(.67, .33))
splitfolders.ratio("lfw_edited", output="lfw_dirs/lfw_edited _predict3", seed=56, ratio=(.50, .50))
splitfolders.ratio("lfw_edited", output="lfw_dirs/lfw_edited _predict4", seed=349, ratio=(.33, .67))
splitfolders.ratio("lfw_edited", output="lfw_dirs/lfw_edited _predict5", seed=90, ratio=(.2, .8))
'''
PARENT_DIR = os.path.dirname(os.path.abspath("__file__")) # path to root directory


# 'Intruder set' -> faces that were not trained on the model
unknown = [x for x in os.listdir('unknown_set') if not x.startswith(".")]

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, keep_all=True
) # MTCNN initialisation for face detection

resnet = InceptionResnetV1(pretrained='vggface2').eval() #Initialisation of InceptionResnetV1 for face recognition


loadImages = torch.load("save_video_model/video_dataset1.pt")

# returns the total number of files
count = 0
for root_dir, cur_dir, files in os.walk('lfw_dirs/lfw_edited_predict5/test'):
    count += len(files)



correct = 0 # number of files correctly recognised
incorrect = 0 # number of files incorrectly recognised
face_detected = 0 # counts the total number of times face has been detected

true_positive = 0 # TPFR: True Positive Face Recognition (Where the face matched with the intendent recipient)
false_positive = 0 # FPFR: False Positive Face Recognition (Where the face matched with the non-intendent recipient)
true_negative = 0 # TNFR: True Negative Face Recognition (Where the unknown face is detected as unknown)
false_negative = 0 # FNFR: False Negative Face Recognition (Where the unknown face is detected as known)
unclassified = 0 # Known faces that fell below the confidence/identification threshold
unclassified_names =[] # stores all the unclassified names


#iterates through test directoryy
for root, dirs, files in sorted(os.walk('lfw_dirs/lfw_edited_predict5/test')):
    for filename in files:
        #skips over .DS_STORE
        if filename.startswith('.'):
            continue
        name = root[34:]
        x = (os.path.join(root, filename))
        read_img = Image.open(x)
        face_cropped, probability = mtcnn(read_img, return_prob=True) #returns cropped image to pass onto resnet
        if face_cropped is not None:
            face_detected += 1
            for i, probability in np.ndenumerate(probability):
                if probability > 0.90:
                    # passes image to process in resnet
                    matrix = resnet(face_cropped)
                    matrix_detach = matrix.detach()
                    saved_data = torch.load('save_lfw_model/lfw_edited_train5.pt')
                    imageEmbeddings = saved_data[0] # contanins the list of all image embeddings stored in .pt file extension
                    name_array = saved_data[1] # contains the list of all name that is associated with the embedding in the .pt file exxtension
                    calcDistance = list() # stores p-norm between the image and the embedding from the image

                    #iterates through .pt embeddings and calculates the p-norm between the input and the output
                    for i, embeddings in enumerate(imageEmbeddings):
                        dist = torch.dist(matrix_detach, embeddings).item()
                        calcDistance.append(dist)

                    low_value = calcDistance.index(min(calcDistance))
                    predict_name = name_array[low_value] #the predict value the machine thinks it is
                    min_dist = min(calcDistance)

        print('Face matched with: ', predict_name, "Distance: ", min(calcDistance), 'Real Identity: ', name)

        if min_dist <= 1:
            if predict_name == name and name not in unknown:
                true_positive += 1
            elif predict_name != name and name not in unknown:
                false_positive += 1
            elif predict_name != name and name in unknown:
                false_negative += 1
        else:
            if predict_name != name and name in unknown:
                true_negative += 1
            else:
                unclassified += 1
                unclassified_names.append(name)

        '''
        if predict_name == name:
            correct += 1
            print('Face correctly matched with: ', predict_name, "Distance: ", min(calcDistance))
        else:
            incorrect +=1
            print('Face incorrectly matched with: ',predict_name, "Distance: ", min(calcDistance))
        '''


#Nuanced Evaluation Metrics
print("Number of faces identified: ", true_positive)
print("Number of faces wrongly identified: ", false_positive)
print("Total number of faces detected: ",face_detected)
print("Total number of images: ", count)
print("Percentage of face detected", (face_detected/count)*100)
print("Percentage of faces recognised:", (true_positive / face_detected) * 100)
print("True Acceptance rate: ", ((true_positive + true_negative)/face_detected)*100)
print("True Negative Face Recognition (Where the unknown face is detected as unknown): ", true_negative)
print("False Negative Face Recognition (Where the unknown face is detected as known): ", false_negative)
print("Faces that were not classified:", unclassified)
most_common_name = max(unclassified_names, key=unclassified_names.count)
print("name: ", most_common_name, "count: ", unclassified_names.count(most_common_name))




#Evaluation metrics to calculate face detection and classification

'''
print("Face correctly identified: ", correct)
print("Face incorrectly identified: ", incorrect)
print("Face Detected :", face_detected/ count*100)
print("Face accuracy: ", (correct/count)*100)
'''


