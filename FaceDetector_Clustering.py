'''
All Implementation.
'''
from helper import show_image

import cv2
import numpy as np
import os
import sys
import face_recognition
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler , StandardScaler
import matplotlib.pyplot as plt
'''
Please do NOT add any imports.
'''

#input_path = "validation_folder/images"

def read_image(actual_path):
    # Reading the Image
    img = cv2.imread(actual_path)
    # Converting image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_image , img
def haar_cascade(gray_image):
    # Cascade Classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Obtaining the face rectangles
    face_rectangles = face_cascade.detectMultiScale(gray_image, scaleFactor = 1.2, minNeighbors = 4)
    return face_rectangles

def haar_cascade_2(gray_image):
    # Cascade Classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Obtaining the face rectangles
    face_rectangles = face_cascade.detectMultiScale(gray_image, scaleFactor = 1.2, minNeighbors = 4)[0]
    return face_rectangles


def detect_faces(input_path: str) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
    for files in os.listdir(input_path):
        img_gray,_ = read_image(f"{input_path}/{files}")
        plt.imshow(img_gray)
        faces_rects = haar_cascade(img_gray)
        
        for x, y, w, h in faces_rects:
            temp_empty_dict ={}
            temp_empty_dict["iname"] = files
            temp_empty_dict["bbox"] = [int(x), int(y), int(w), int(h)]
            result_list.append(temp_empty_dict)
    return result_list

def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    '''
    Your implementation.

    '''
    feature_list = []
    image_name_list = []
    for files in os.listdir(input_path):
        #print(f"{input_path}/{files}")
        img_gray , img_color = read_image(f"{input_path}/{files}")
        #plt.imshow(img_gray)
        faces_rects = haar_cascade_2(img_gray)
        # Loading the required haar-cascade xml classifier file
        x, y, w, h = faces_rects
        #print(x, y, w, h,)
        image_cropped = img_color[y:y+h,x:x+w]
        encoding = face_recognition.face_encodings(image_cropped)
        try:
            feature_list.append(encoding[0])
            image_name_list.append(files)
        except Exception:
            pass
    numpy_feature_array = np.array(feature_list)
    #print(numpy_feature_array)

    # Scaling the array before  
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(numpy_feature_array)
    # Kmeans Clustering 
    kmeans_model=KMeans(n_clusters=int(K))
    kmeans_predict = kmeans_model.fit_predict(scaled_array)
    zipped_imgname_clusters =list(zip(image_name_list,kmeans_predict))
    for k_cluster in range(int(K)):
        img_group_list = []
        temp_empty_dict = {}
        for tups in zipped_imgname_clusters:
            if tups[1]==k_cluster:
                img_group_list.append(tups[0])
            temp_empty_dict["cluster_no"] = k_cluster
            temp_empty_dict["elements"] = img_group_list
        result_list.append(temp_empty_dict)
    

    return result_list
#rl = cluster_faces("faceCluster_5",int(5))

#print(rl)
