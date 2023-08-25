from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import os
from PIL import Image
import torch
import time
import numpy as np

#threshold -> 0.9, 1.2, 1.5

select_test_folder = 5

#__________________________ code for calculating everything_________________________

# unregistered set
intruder_set = ['sandra_bullock','sarah_mclachlan', 'tony_blair', 'gloria_estefan', 'jennifer_aniston', 'joe_torre', 'steven_spielberg', 'orlando_bloom', 'matt_damon', 'sylvester_stallone']

#initialising MTCNN face detection
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
)

#initialising InceptionResnetV1
resnet = InceptionResnetV1(pretrained='vggface2').eval()

path = "videoFolders/videoFolder_ten" + str(select_test_folder)

face_detected = 0
total_frames =0
total_true_positives = 0
total_false_positives = 0
total_unclassified =0
total_true_negatives = 0
total_false_negatives = 0

loadImages = torch.load('save_video_model/video_dataset{number}.pt'.format(number = select_test_folder))
imageEmbeddings, loadNames = loadImages[0], loadImages[1]
pastTimeFrame = 0
newTimeFrame = 0
def findName(calcDistance):
    minValue = min(calcDistance.items(), key=lambda x: x[1])
    val_pos = minValue[0]
    minimumDistance = minValue[1]
    name = loadNames[val_pos]
    return minimumDistance, name


def findMinDist(calcDistance):
    for idx, embeddingDB in enumerate(imageEmbeddings):
        dist = torch.dist(matrix, embeddingDB)
        matrixDist = dist.item()
        calcDistance[idx] = matrixDist

for file in os.listdir(path):
    if file.startswith('.'):
        continue
    print(file)

    cam = cv2.VideoCapture(path + '/' + file)
    count = 0
    true_positive = 0
    false_positive = 0
    unknown = 0
    true_negative = 0
    false_negative = 0
    unclassified = 0
    while True:
        ret, frame = cam.read()
        if ret == True:

            image = Image.fromarray(frame)
            imageAlignment, probabilityArray = mtcnn(image, return_prob=True)

            if imageAlignment is not None:
                boundingBoxes, _, landmarks = mtcnn.detect(image, landmarks=True)
                for i, probability in np.ndenumerate(probabilityArray):

                    if probability > 0.90:
                        face_detected += 1
                        matrix = resnet(imageAlignment[i].unsqueeze(0))
                        embeddings = matrix.detach()

                        calcDistance = dict()

                        findMinDist(calcDistance)
                        minimumDistance, name = findName(calcDistance)

                        if minimumDistance < 1.05:

                            if name == file[12:].replace(".avi","") and file[12:].replace(".avi","") not in intruder_set:
                                print("identity detected: " + name + " actual identity: " + file[12:].replace(".avi",""))
                                true_positive += 1
                            elif name != file[12:].replace(".avi","") and file[12:].replace(".avi","") not in intruder_set:
                                print("False identity! Resnet predicts: " + name + " actual identity: " + file[12:].replace(".avi",""))
                                false_positive += 1
                            elif name != file[12:].replace(".avi","") and file[12:].replace(".avi","") in intruder_set:
                                false_negative += 1
                        else:
                            if name != file[12:].replace(".avi","") and file[12:].replace(".avi","") in intruder_set:
                                true_negative += 1
                            else:
                                unclassified += 1



        if not ret:
            break


        # process frame
        count += 1
        total_frames += 1
    total_true_positives += true_positive
    total_false_positives += false_positive
    total_true_negatives += true_negative
    total_false_negatives += false_negative
    total_unclassified += unclassified
    
    print(f"{count} frames read")
    print(f"{true_positive} true positives detected")
    print(f"{false_positive} false positives detected")
    print(f"{true_negative}true negatives detected")
    print(f"{false_negative}false negatives detected")
    print(f"{unclassified}unknown detected")




    cam.release()

print("Total frames in the  directory: ", total_frames)
print("Total face detected in frames: ",  face_detected)
print("MTCNN detected: ", face_detected/total_frames *100, "% of faces from total number of faces")
print("Total number of true_positives: ", total_true_positives)
print("Total number of false_positives: ", total_false_positives)
print("Total number of true_negatives: ", total_true_negatives)
print("Total number of false_negatives: ", total_false_negatives)
print("Total number of unclassified: ", total_unclassified)
total = total_true_positives + total_false_positives + total_false_negatives + total_true_negatives + total_unclassified
print("True Positive percentage: ", (total_true_positives/ total)*100)
print("True Acceptance percentage: ", ((total_true_positives + total_true_negatives)/ total)*100)


