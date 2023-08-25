import random

import torchvision
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import PIL
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import time
from torchvision import datasets
from torch.utils.data import DataLoader
import os

import cv2


class CamApp(App):

    def build(self):
        return self.home()
    
    def home(self):
        self.img1 = Image()
        self.txt = TextInput(text='Enter Name:', size_hint=(1, .1))
        self.button2 = Button(text="Take photo", on_press=self.takePicture, size_hint=(1, .1))
        #self.button3 = Button(text="play", on_press=self.play, size_hint=(1, .1))
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img1)
        layout.add_widget(self.txt)
        layout.add_widget(self.button2)
        #layout.add_widget(self.button3)
        self.capture = cv2.VideoCapture(0)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        FPS = 33.0
        Clock.schedule_interval(self.update, 1.0 / FPS)
        return layout


    def update(self, dt):
        # display image from cam in opencv window
        mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40)  # keep_all=True

        # matches face to name
        def findName(calcDistance):
            minValue = min(calcDistance.items(),
                           key=lambda x: x[1])  # returns key-value with the lowest embedding value
            val_pos = minValue[0]
            minimumDistance = minValue[1]
            name = loadNames[val_pos]
            return minimumDistance, name

        # calculates min distance
        def findMinDist(calcDistance):
            # iterates through the pre-saved embeddings
            for i, emb in enumerate(imageEmbeddings):
                dist = torch.dist(matrix, emb)
                matrixDist = dist.item()
                calcDistance[i] = matrixDist  # adds embedding to dict

        if os.path.isfile('app_data.pt'):
            loadImages = torch.load('app_data.pt')
            imageEmbeddings, loadNames = loadImages[0], loadImages[1]

            ret, frame = self.capture.read()
            if ret == True:
                image = PIL.Image.fromarray(frame)
                imageAlignment, probabilityArray = mtcnn(image, return_prob=True)
                if imageAlignment is not None:
                    gamma_correction = torchvision.transforms.functional.adjust_gamma(image,0.7)  # pre-processing for face recognition in low illuminated spaces
                    boundingBoxes, _, landmarks = mtcnn.detect(image, landmarks=True)
                    for i, probability in np.ndenumerate(probabilityArray):
                        if probability > 0.9:
                            matrix = self.resnet(imageAlignment[i].unsqueeze(0))
                            matrix.detach()
                            calcDistance = dict()
                            findMinDist(calcDistance)
                            minimumDistance, name = findName(calcDistance)
                            for i, (box, point) in enumerate(zip(boundingBoxes, landmarks)):
                                x = round(box[0])
                                y = round(box[1])
                                w = round(box[2])
                                h = round(box[3])
                                if minimumDistance < 0.9:
                                    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                                cv2.LINE_AA)
                                else:
                                    cv2.putText(frame, 'unknown', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                                cv2.LINE_AA)
                                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 3)
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = texture1

        else:
            ret, frame = self.capture.read()
            self.datasetupdate()
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = texture1


    def takePicture(self, *args):
        PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
        IMG_DIR = os.path.join(PARENT_DIR, 'app_image')
        name = self.txt.text
        save_name = name.title().replace(" ", "_")
        new_path = os.path.join(IMG_DIR, save_name)
        img_counter = 0
        if not os.path.exists(new_path):
            print("making directory...")
            os.mkdir(new_path)
            for i in range(10):
                ret, frame = self.capture.read()
                img_name = new_path + "/{}_{}.jpg".format(save_name, i)
                cv2.imwrite(img_name, frame)
                img_counter += 1
        self.datasetupdate()


    def datasetupdate(self, *args):
        mtcnn = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40)  # keep_all=False

        dataset = datasets.ImageFolder('app_image')  # photos folder path
        label_ids = dict()  # stores name and id as key-value pairs
        current_id = 0
        BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
        IMAGE_DIR = os.path.join(BASE_DIR, 'app_image')

        # iterates through directories | sub-directories | image files
        for root, dirs, files in sorted(os.walk(IMAGE_DIR)):
            for file in files:
                # only loads file with JPEG extension
                if file.endswith("jpg"):
                    path = os.path.join(root, file)
                    path_name = os.path.dirname(path)
                    # stores the name of the person
                    label = os.path.basename(path_name)
                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id += 1

        # reverses key-value pairs so current
        dataset.labels = {v: k for k, v in label_ids.items()}

        # stores pictures in the PIL format
        loader = DataLoader(dataset, collate_fn=lambda x: x[0])

        name_array = list()  # list of names corrosponding to cropped photos
        embedding_array = list()  # stores embedding after image crop from mtcnn

        # images and its corresponding index value is looped through the loader
        for image, index in loader:
            # returns cropped face and probability of it
            face, prob = mtcnn(image, return_prob=True)
            if face is not None:
                # pass the face into the pre-trained model and unsqueeze is because resnet needs 4 dimensions and returns embedding which is appended onto the empty array
                convertForModel = face.unsqueeze(0)
                matrix = self.resnet(convertForModel)
                # detach releases memory
                embedding_array.append(matrix.detach())
                # the name is saved onto name list
                name_array.append(dataset.labels[index])

        # save data
        celebFaceDB = list((embedding_array, name_array))
        torch.save(celebFaceDB, 'app_data.pt')
        del mtcnn




if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()




