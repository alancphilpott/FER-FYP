from keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

# Start Dataset Helper Functions #
def load_dataset():
    data = pd.read_csv('../dataset/fer2013.csv') # read dataset file into Pandas DataFrame
    all_images_pixels = data['pixels'].tolist() # array of each images pixel data, in a list, comma seperated
    width, height = 48, 48 # dimensions of dataset image
    input_shape = (64, 64, 1) # network image input shape
    faces = []
    for images_pixels in all_images_pixels: # for each images pixels
        face_image = [int(pixel) for pixel in images_pixels.split(' ')] # create a unique face of pixels
        face_image = np.asarray(face_image).reshape(width, height) # reshape image to 48x48 array
        face_image = cv2.resize(face_image.astype('uint8'), input_shape[:2]) # resize the image to what network expects
        faces.append(face_image.astype('float32')) # add face with value types as float32
    faces = np.asarray(faces) # convert faces to an array
    faces = np.expand_dims(faces, -1) # reduce the shape of the faces array by 1
    emotions = pd.get_dummies(data['emotion']).as_matrix() # convert emotion values into dummy variables
    data = faces, emotions # assign and return the data
    return data

def get_labels():
    return {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}

def split_data(faces, emotions, validation_split=.2):
    total_faces = len(faces) # total number of face samples
    num_train_samples = int((1 - validation_split)*total_faces) # assign 80% of the samples as training samples
    train_faces = faces[:num_train_samples] # new array of faces for training
    train_emotions = emotions[:num_train_samples] # new array of emotions for training
    val_faces = faces[num_train_samples:] # new array of faces for validation
    val_emotions = emotions[num_train_samples:] # new array of emotions for validation
    train_data = (train_faces, train_emotions) # training data
    val_data = (val_faces, val_emotions) # validation data
    return train_data, val_data # return the split data
# End Dataset Functions #

# Start Image Helper Functions #
def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
# End Image Helper Functions #

# Start Function #
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
# End Function #