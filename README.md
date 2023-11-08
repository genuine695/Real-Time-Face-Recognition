# Real-Time Face Recognition System

## Project Overview

This repository contains the implementation of my final year project, a real-time face recognition system. The objective of this system is to accurately discern, detect, and identify known faces from a given set of images. It labels recognized faces with their names within a bounding box and tags unrecognized individuals as unknown. This project marks my initial foray into the field of machine learning and has successfully achieved a true recognition rate of over 76% under varying conditions.

## Key Features

- **Real-Time Detection**: The system can detect and identify faces in real-time.
- **Identification of Known Faces**: Known faces are recognized and labelled with corresponding names.
- **Handling Unknown Faces**: Faces that are not trained in the system are displayed as unknown.
- **Performance**: Achieved a significant true recognition rate in diverse conditions.

## Research and Development

A thorough literature review was conducted to understand various face detection and recognition algorithms, industry-standard datasets, and the challenges inherent in facial recognition technology. The decision to implement the state-of-the-art facenet model was made after considering several potential solutions. This model integrates:

- **MTCNN for Face Detection**: Utilizes a Multi-task Cascaded Convolutional Networks (MTCNN) framework for accurate face detection.
- **Deep Convolutional Networks for Recognition**: Employs deep learning for feature extraction and recognition.

## Tools and Technologies

- **Python**: For writing the core algorithms and handling data processing.
- **OpenCV**: Used for image processing and real-time video analytics.
- **Albumentations**: An augmentation library to enhance model performance by simulating various conditions.
- **PyTorch**: A machine learning framework that accelerates the path from research prototyping to production deployment.
- **KivyApp**: For creating a user-friendly interface for the application.

## Results

The system was rigorously tested and was able to recognize faces with a high accuracy rate in different scenarios, proving the effectiveness of the facenet model coupled with MTCNN.
