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
CK+ Dataset contains multile directory.
Most of this directory's contain set of images and a text file.
The Images are of actor's face, from their normal face till a particular emotion.
The text file contains a emotional label(a number) for the particular emotion.
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
            python %(prog)s /home/user/datasets/ck_dataset
            python %(prog)s /home/user/datasets/ck_dataset --outfile face_datset.pickle
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
parser.add_argument('--extractFace', dest='detect_face', action="store_true",
                    help='Crop and save face in the img')
parser.add_argument('--resize', nargs=2, type=int, default=[100, 100],
                    help='Resize image to paticular dimensions (w x h) ')

[dataset_path, outfile, training_size, validation_size, testing_size, detect_face, resize] = vars(parser.parse_args()).values()
resize = tuple(resize)

if training_size + validation_size + testing_size != 100:
    raise  argparse.ArgumentTypeError(
        "Training Size, Validation Size, Testing Size should be equal be 100"
    )

if not os.path.exists(dataset_path):
    raise IOError('No such file or directory', dataset_path)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
ck_dataset = 8 * [[]]


def detect_face_resize(imgpath):
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    if detect_face:
        faces = face_cascade.detectMultiScale(img, 1.25, 5)
        c = 1.25
        while not len(faces):
            faces = face_cascade.detectMultiScale(img, c, 5)
            c -= .05
            if c <= 1:
                print(imgpath) # Change parameters of detectMultiScale or manually crop the image
                break
        for (x,y,w,h) in faces:
            cv2.imwrite(imgpath, cv2.resize(img[y:y+h, x:x+w], resize, interpolation = cv2.INTER_AREA))
    else:
        cv2.imwrite(imgpath, cv2.resize(img, resize, interpolation = cv2.INTER_AREA))

def trav_dir(dataset_path):
    gen = os.walk(dataset_path)
    next(gen)
    # for creating seprate directory with
    # prefix = "_copy"
    # while(!os.path.exists(dataset_path + prefix)):
    #     prefix += "_copy"
    # os.mkdir(dataset_path + prefix)
    for root, dirs, files in gen:
        # os.mkdir(os.path.join(dataset_path + prefix, files))
        for file in files:
            if(imghdr.what(os.path.join(root, file)) in ['png']): # why in used why not ==
                detect_face_resize(os.path.join(root, file))

def crop_and_resize(dataset_path):
    trav_dir(dataset_path)

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
            img = cv2.imread(file,cv2.IMREAD_GRAYSCALE).flatten() #changes image matrix to single dimension in a row major fashion
            img = (img/25500).astype(np.float32) # normalization of data [for sigmoid neurons(0-1)]
            if int(file[-12:-4]) <= threshold: #Image name are of type 'S005_001_00000002.png' looks at the part '00000002'
                ck_dataset[0].append(img)
            else:
                ck_dataset[emotion_label].append(img)

def trav_dir_1(dataset_path):
    gen = os.walk(dataset_path)
    next(gen)
    # for creating seprate directory with
    # prefix = "_copy"
    # while(!os.path.exists(dataset_path + prefix)):
    #     prefix += "_copy"
    # os.mkdir(dataset_path + prefix)
    for root, dirs, files in gen:
        # os.mkdir(os.path.join(dataset_path + prefix, files))
        if not any(".txt" in file for file in files): #check whether directory have emotion label
            continue
        files_path = []
        for file in files:
            files_path.append(os.path.join(root, file))
        load_data(files_path)

def pack_data():
    global ck_dataset

    training_data = []
    validation_data = []
    test_data = []

    training_txt = []
    validation_txt = []
    test_txt = []

    for x,y in zip(ck_dataset,range(0,8)):
        i1, i2 = math.ceil(len(x)*0.8), math.floor(len(x)*0.9)

        training_data.append(x[0:i1])       #80 percent Image of each emotion for training
        validation_data.append(x[i1:i2])    #10 percent
        test_data.append(x[i2:len(x)])      #10 percent

        training_txt.append(y*np.ones(shape=(len(x[0:i1]),1), dtype=np.int8)) #Generating Corresponding Label
        validation_txt.append(y*np.ones(shape=(len(x[i1:i2]),1), dtype=np.int8))
        test_txt.append(y*np.ones(shape=(len(x[i2:len(x)]),1), dtype=np.int8))

    #np.vstack(list_of_array) converts the list_of_numpy_arrays into a single numpy array
    training_data, validation_data, test_data = np.vstack(training_data), np.vstack(validation_data), np.vstack(test_data)
    training_txt, validation_txt, test_txt = np.vstack(training_txt), np.vstack(validation_txt), np.vstack(test_txt)

    with open(outfile, 'wb') as f:
        pickle.dump({
            "training_data"   : [ training_data, training_txt],
            "validation_data" : [ validation_data, validation_txt],
            "test_data"       : [ test_data, test_txt],
            "img_dim"         : {"width": resize[0], "height": resize[1]}
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

def serialize_CK(dataset_path):
    trav_dir_1(dataset_path) #name of the directory
    pack_data()

crop_and_resize(dataset_path)
serialize_CK(dataset_path)
