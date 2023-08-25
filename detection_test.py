from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import os


# ------------code to check how many of the frames face recognition acc  took place--------------

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
)

path = "testVidDirectory"
face_detected = 0
total_frames = 0
cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

for file in os.listdir(path):
    if file.startswith('.'):
        continue
    print(file)

    cap = cv2.VideoCapture(path + '/' + file)
    count = 0

    while True:
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            original_frame = frame.copy()
            for (x, y, w, h) in faces:
                face_detected += 1

            # image = Image.fromarray(frame)
            # imageAlignment, probabilityArray = mtcnn(image, return_prob=True)
            # if imageAlignment is not None:
            #    boundingBoxes, _, landmarks = mtcnn.detect(image, landmarks=True)
            #    face_detected += 1

        # check for end of file

        if not ret:
            break

        # process frame
        count += 1
        total_frames += 1

    print(f"{count} frames read")

    cap.release()

print(total_frames)
print(face_detected)
print(face_detected / total_frames * 100)