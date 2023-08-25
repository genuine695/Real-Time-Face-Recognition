import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import cv2
import time
import numpy as np
import os
from torch.utils.data import DataLoader

#-----------------------returns path to directory | files | images--------------------

PARENT_DIR = os.path.dirname(os.path.abspath("__file__")) # path to root directory
IMAGE_DIR = os.path.join(PARENT_DIR,'train_image') # path to train_image directory
DATA_PATH = os.path.join(PARENT_DIR, 'saveEmbeddings')
TRAIN_EMBEDS = os.path.join(DATA_PATH, 'trainEmbeds.npz')
weights = os.path.join(PARENT_DIR, "face_detection_yunet_2022mar.onnx")
dnn_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))


mtcnn = MTCNN(
    image_size=160, margin=0, keep_all=True,
    min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709
) # initialises MTCNN(MultiCascade Neural Network) for face detection

resnet = InceptionResnetV1(pretrained='vggface2').eval() # initialises InceptionResnetV1 for face recognition

loadImages = torch.load('train_image2.pt') # loads file containing name and embeddings
imageEmbeddings, loadNames = loadImages[0], loadImages[1]

#fps = 1/pastTimeFrame - newTimeFrame
pastTimeFrame = 0
newTimeFrame = 0

# webcam initialisation
cam = cv2.VideoCapture(0)


# saves video as 'output.mp4' file at the end
fourcc = cv2.VideoWriter_fourcc(*'XVID')
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))
img_counter = 0

# saves name and embeddings as a .pt file
def dataset():

    label_ids = dict() # stores name and id as key-value pairs
    current_id = 0

    #iterates through directories | sub-directories | image files
    for root, dirs, files in sorted(os.walk(IMAGE_DIR)):
        for file in files:
            if file.endswith("jpg"):
                path = os.path.join(root, file)
                path_name = os.path.dirname(path)
                label = os.path.basename(path_name) # returns directory name as label
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

    #reverses key-value pairs so current
    dataset.labels = {v: k for k, v in label_ids.items()}
    print(dataset.labels)

    # stores pictures in the PIL format
    loader = DataLoader(dataset, collate_fn=lambda x: x[0])

    name_array = list()  # list of names corrosponding to cropped photos
    emb_array = list()  # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

    # images and its corresponding index value is looped through the loader
    for image, index in loader:
        # returns cropped face and probability of it
        face, prob = mtcnn(image, return_prob=True)
        if face is not None:
            # pass the face into the pre-trained model and unsqueeze is because resnet needs 4 dimensions and returns embedding which is appended onto the empty array
            convertForModel = face.unsqueeze(0)
            embedding = resnet(convertForModel)
            # detach releases memory
            emb_array.append(embedding.detach())
            # the name is saved onto name list
            name_array.append(dataset.labels[index])

        # save data as numpy array
    np.savez(TRAIN_EMBEDS, x=emb_array, y=name_array)

    # save data as PT extension
    celebFaceDB = list((emb_array, name_array))
    torch.save(celebFaceDB, 'train_image2.pt')


# matches face to name
def findName(calcDistance):
    minValue = min(calcDistance.items(), key=lambda x: x[1]) # returns key-value with the lowest embedding value
    val_pos = minValue[0]
    minimumDistance = minValue[1]
    name = loadNames[val_pos]
    return minimumDistance, name

# calculates min distance
def findMinDist(calcDistance):
    #iterates through the pre-saved embeddings
    for i, emb in enumerate(imageEmbeddings):
        dist = torch.dist(matrix, emb)
        matrixDist = dist.item()
        calcDistance[i] = matrixDist # adds embedding to dict


while True:
    ret, frame = cam.read()
    newTimeFrame = time.time()
    if ret == True:
        image = Image.fromarray(frame) # copies each frame and converts to PIL image format for facial recognition
        imageAlignment, probabilityArray = mtcnn(image, return_prob=True)
        if imageAlignment is not None:
            gamma_correction = torchvision.transforms.functional.adjust_gamma(image, 0.7) # pre-processing for face recognition in low illuminated spaces
            boundingBoxes, _, landmarks = mtcnn.detect(image, landmarks=True)
            for i, probability in np.ndenumerate(probabilityArray):
                if probability > 0.90:

                    #unsqueeze adds a batch dimension
                    matrix = resnet(imageAlignment[i].unsqueeze(0))
                    matrix.detach()
                    calcDistance = dict()
                    findMinDist(calcDistance)
                    minimumDistance, name = findName(calcDistance)

                    # duplicate frame to take images if needed
                    copy_frame = frame.copy()


                    # bounding box
                    for i, (box, point) in enumerate(zip(boundingBoxes, landmarks)):
                        x = int(box[0])
                        y = int(box[1])
                        w = int(box[2])
                        h = int(box[3])
                        if minimumDistance < 0.90:
                            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) # display name in webcam
                        else:
                            cv2.putText(frame, "unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                        cv2.LINE_AA)
                        # displays bounding box in webcam
                        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 3)

                        # displays facial landmarks (left-eye, right-eye, two corners of lips and nose)
                        cv2.circle(frame, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), 3)
                        cv2.circle(frame, (int(point[1][0]), int(point[1][1])), 3, (0, 255, 0), 3)
                        cv2.circle(frame, (int(point[2][0]), int(point[2][1])), 3, (0, 255, 0), 3)
                        cv2.circle(frame, (int(point[3][0]), int(point[3][1])),  3, (0, 255, 0), 3)
                        cv2.circle(frame, (int(point[4][0]), int(point[4][1])), 3, (0, 255, 0), 3)



        #calculates fps
        fps = int(1 / (newTimeFrame - pastTimeFrame))
        pastTimeFrame = newTimeFrame
        cv2.putText(frame, str(fps), (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # saves video capture from webcam as a file
        out.write(frame)

        # displays webcam
        cv2.imshow("webcam", frame)

        #shuts down facecam if pressed ESC
        if cv2.waitKey(1) % 256 == 27:
            print('Shutting webcam....')
            break

        #takes 10 pictures if pressed space
        if cv2.waitKey(1) % 256 == 32:
            name = input("Enter name:\n")
            name = name.title().replace(" ", "")
            print(name)

            # makes directory if it name does not exist
            try:
                new_path = os.path.join(IMAGE_DIR, name)
                if not os.path.exists(new_path):
                    print("making directory...")
                    os.mkdir(new_path)
                    for i in range(10):
                        img_name = new_path + "/{}_{}.jpg".format(name, i)
                        #img_name = new_path + "/{}_{}.jpg".format(name.title().replace(" ", "_"), i)
                        cv2.imwrite(img_name, copy_frame)
                        img_counter += 1
                    dataset()
                else:
                    print("error: already exists")
            except:
                print("exception raised")

cam.release()
out.release()
cv2.destroyAllWindows() # display window
cv2.waitKey(1)

del mtcnn
del resnet





