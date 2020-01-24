# FER

Final Year Project May 2019
- An application created to detect facial emotion both through images and in real-time using a webcam.

# Dataset Info
"The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)."

The training set consists of 28,709 images displaying emotions. The images are loaded and preprocessed prior to training the model utilizing a number of functions within helper.py. 80% of the images are used for training, while 20% for validation. DUring training, the most accurately resulting model at the time is saved.

The models saved here are the most accurate, utilizing the XCEPTION architecture.

The application then loads the face detection model and trained emotion model used in the prediction process.
