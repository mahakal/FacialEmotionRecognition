import os
#for imghdr.what to find whether a file is image
#['pbm','pgm','ppm','png','jpeg','tiff','bmp','webp']
import imghdr
import cv2
import pickle
import math
import numpy as np
import argparse

'''
CK Dataset contains multiple directory.
Most of this directory's contain one or more directory's, with each of
the directory containing a set of images and a text file.
The Images are of actor's face, from their normal face till a particular emotion.
The text file contains a emotional label(a number) for the particular emotion.
Emotion Label for corresponding Emotional Expression
0:'neutral', 1:'anger', 2:'contempt', 3:'disgust', 4:'fear', 5:'happy', 6:'sadness', 7:'surprise'
'''

if __name__ != '__main__':
    raise ImportError('Should be run as Script')

parser = argparse.ArgumentParser(
    description='''
        Serialize\'s CK+ Dataset into pickle format.
    ''',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    epilog='''
        Examples:
            python %(prog)s /home/user/datasets/ck_dataset --crop
            python %(prog)s /home/user/datasets/ck_dataset --outfile ck_dataset.pickle
            python %(prog)s /home/user/datasets/ck_dataset --training 70 --validation 20 --test 10
            python %(prog)s /home/user/datasets/ck_dataset -t 70 -v 20 -test 10 -o fd.pickle

            Note: Training Size, Validation Size, Testing Size should be equal be 100
    '''
)

parser.add_argument('dataset_path', help='Absolute Path of the CK+ Dataset')
parser.add_argument('-o', '--outfile', default='ck_dataset.pickle',
                    help='Name of the output pickle file')
parser.add_argument('-t', '--training', dest='training_size', type=int, default=80,
                    help='Percent of dataset to use for Training')
parser.add_argument('-v', '--validation', dest='validation_size', type=int,
                    default=10, help='Percent of dataset to use for Validation')
parser.add_argument('-test', '--test', dest='testing_size', type=int, default=10,
                    help='Percent of dataset to use for Testing')
parser.add_argument('--crop', dest='detect_face', action="store_true",
                    help='Crop and save face in the img')
parser.add_argument('--resize', nargs=2, type=int, default=[100, 100],
                    help='Resize image to paticular dimensions (w x h) ')

dataset_path, outfile, training_size, validation_size, testing_size, detect_face, resize = vars(parser.parse_args()).values()

if training_size + validation_size + testing_size != 100:
    raise  argparse.ArgumentTypeError(
        "Training Size, Validation Size, Testing Size should be equal be 100"
    )

if not os.path.exists(dataset_path):
    raise IOError('No such file or directory', dataset_path)

training_size, validation_size, testing_size, resize = training_size/100, validation_size/100, testing_size/100, tuple(resize)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
ck_dataset = [[], [], [], [], [], [], [], []]

def detect_face_resize(imgpath):
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    if detect_face:
        faces = face_cascade.detectMultiScale(img, 1.25, 5)
        while not len(faces):
            scale_factor = 1.20
            faces = face_cascade.detectMultiScale(img, scale_factor, 5)
            scale_factor -= .05
            if scale_factor <= 1:
                print(imgpath) # Change parameters of detectMultiScale or manually crop the image
                return
        for (x,y,w,h) in faces:
            return cv2.resize(img[y:y+h, x:x+w], resize, interpolation = cv2.INTER_AREA)
    else:
        return cv2.resize(img, resize, interpolation = cv2.INTER_AREA)

def load_data(files_path):
    global ck_dataset

    threshold = math.floor( (len(files_path)-1)*0.3) #how many pictures be considered neutral in directory

    for file in files_path: #find emotion label and read it
        if ".txt" in file:
            with open(file, 'rb') as text_file:
                emotion_label = int(float(text_file.readline()))
            break

    for file in files_path:
        if imghdr.what(file) in ['png']: #Makes sure file is .png image
            img = detect_face_resize(file)
            if img is None:
                continue
            img = img.flatten()
            img = (img/max(img)).astype(np.float32) # normalization of data [for sigmoid neurons(0-1)]
            if int(file[-12:-4]) <= threshold: #Image name are of type 'S005_001_00000002.png' looks at the part '00000002'
                ck_dataset[0].append(img)
            else:
                ck_dataset[emotion_label].append(img)

def trav_dir(dataset_path):
    gen = os.walk(dataset_path)
    next(gen)
    for root, dirs, files in gen:
        if dirs and not any(".txt" in file for file in files): #check whether directory have emotion label
            continue
        files_path = []
        for file in files:
            files_path.append(os.path.join(root, file))
        load_data(files_path)

def serialize_CK(dataset_path):
    trav_dir(dataset_path)

    training_data = []
    validation_data = []
    test_data = []

    training_label = []
    validation_label = []
    test_label = []

    for x,y in zip(ck_dataset,range(0,8)):
        i1, i2 = math.ceil(len(x)*training_size), math.floor(len(x)*(training_size+validation_size))

        training_data.append(x[0:i1])
        validation_data.append(x[i1:i2])
        test_data.append(x[i2:len(x)])

        training_label.append(y*np.ones(shape=(len(x[0:i1]),1), dtype=np.int8)) #Generating Corresponding Label
        validation_label.append(y*np.ones(shape=(len(x[i1:i2]),1), dtype=np.int8))
        test_label.append(y*np.ones(shape=(len(x[i2:len(x)]),1), dtype=np.int8))

    #np.vstack(list_of_array) converts the list_of_numpy_arrays into a single numpy array
    training_data, validation_data, test_data = np.vstack(training_data), np.vstack(validation_data), np.vstack(test_data)
    training_label, validation_label, test_label = np.vstack(training_label), np.vstack(validation_label), np.vstack(test_label)

    with open(outfile, 'wb') as f:
        pickle.dump({
            "training_data"   : [ training_data, training_label],
            "validation_data" : [ validation_data, validation_label],
            "test_data"       : [ test_data, test_label],
            "img_dim"         : {"width": resize[0], "height": resize[1]}
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

serialize_CK(dataset_path)
