import time

import PIL
from facenet_pytorch import MTCNN
import cv2
import os



#----------------------tested for iterates through everything------------------------------

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
) # initialises MTCNN for face detection

#path to dnn file detection directory
directory = os.path.dirname(__file__)
weights = os.path.join(directory, "face_detection_yunet_2022mar.onnx")
dnn_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))

# reads video file
testvid = 'testVidDirectories/testVidDirectory1'
total_frames =0

def boundingBox():
    # Draw bounding boxes and landmarks for detected faces
    for face in faces:
        box = list(map(int, face[:4]))
        print(box)
        cv2.rectangle(frame, box, (0, 0, 255), 2, cv2.LINE_AA)


#path to viola-jones face detection algorithm
cascPath= os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
#faceCascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")

previous_time = 0
new_time =0

face_detected = 0
high_fps_list = []
average_list = []
for file in sorted(os.listdir(testvid)):
    if file.startswith('.'):
        continue
    cap = cv2.VideoCapture(testvid + "/" + file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    individual_fps_list = []
    while (cap.isOpened()):
        # Capture frame-by-frame
        new_time = time.time()
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                face_detected += 1


        #    image = PIL.Image.fromarray(frame)
        #    imageAlignment, probabilityArray = mtcnn(image, return_prob=True)
        #    if imageAlignment is not None:
        #        boundingBoxes, _, landmarks = mtcnn.detect(image, landmarks=True)
        #        for i, (box, point) in enumerate(zip(boundingBoxes, landmarks)):
        #            face_detected + =1
        #            cv2.rectangle(frame, (list(map(int, box[:2]))), (list(map(int, box[2:4]))), (0, 255, 0), 3)

            #h, w, _ = frame.shape
            #dnn_detector.setInputSize((w, h))

            # face detection

            #_, faces = dnn_detector.detect(frame)
            #if faces is not None:
            #    faces = faces
            #else:
            #    faces = []

            #boundingBox()

            fps = 1 / (new_time - previous_time)
            previous_time = new_time
            fps = int(fps)
            if individual_fps_list is None and fps != 0:
                individual_fps_list.append(fps)
            elif fps not in individual_fps_list and fps != 0:
                individual_fps_list.append(fps)

            cv2.putText(frame,str(fps) , (8,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0 ), 2)
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break
        total_frames +=1

    high_fps_list.append(individual_fps_list)
    average_list.append(int(sum(individual_fps_list) / len(individual_fps_list)))
print(length)
print("Highest FPS: ", max(individual_fps_list))
print("Lowest FPS: ", min(individual_fps_list))
print('Average FPS: ', int(sum(individual_fps_list) / len(individual_fps_list)))
print("face detected percentage: ", face_detected/total_frames*100)
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()


i=0
file_list = [x for x in os.listdir(testvid) if not x.startswith(".")]
for x in sorted(file_list):
    print("filename: ", x[12:].replace(".avi",""), "Lowest FPS: ", min(high_fps_list[i]), "Highest FPS: ", max(high_fps_list[i]), "Average: ", average_list[i])
    i += 1
    
