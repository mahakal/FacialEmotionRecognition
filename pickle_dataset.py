import os
import imghdr
import cv2
import pickle
import math
import numpy as np
import argparse


'''
CK+ Dataset
'''

parser = argparse.ArgumentParser(
    description='Serialize\'s CK+ Dataset into pickle format.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('dataset_path', help='Absolute Path of the CK+ Dataset')
parser.add_argument('--outfile', default='ck_dataste.pickle',
                    help='Name of the output pickle file')
parser.add_argument('--training', dest='training_size', type=int, default=80,
                    help='Percent of dataset to use for Training')
parser.add_argument('--validation', dest='validation_size', type=int,
                    default=10, help='Percent of dataset to use for Validation')
parser.add_argument('--test', dest='testing_size', type=int, default=10,
                    help='Percent of dataset to use for Testing')

[dataset_path, outfile, training_size, validation_size, testing_size] = vars(parser.parse_args()).values()

ck_dataset = {
    'neutral_img': [],
    'anger_img': [],
    'contempt_img': [],
    'disgust_img': [],
    'fear_img': [],
    'happy_img': [],
    'sadness_img': [],
    'surprise_img': [],
}

'''
traverse directory tree if a directory found with
image then find if .txt file exist if exist and
start collecting data
'''
def load_data(list):
    global ck_dataset
    if any(".txt" in s for s in list): #check whether directory have emotion label

        threshold = math.floor( (len(list)-1)*0.3) #how many pictures be considered neutral in directory

        for x in list: #find emotion label and read it
            if ".txt" in x:
                with open(x, 'rb') as text_file:
                    text = int(float(text_file.readline()))
                break

        for x in list:
            if imghdr.what(x) in ['png']: #Makes sure file is .png image
                img = cv2.imread(x,cv2.IMREAD_GRAYSCALE).flatten() #changes image matrix to single dimension in a row major fashion
                img = (img/25500).astype(np.float32) # normalization of data [for sigmoid neurons(0-1)]
                if int(x[10:-4]) <= threshold: #Image name are of type 'S005_001_00000002.png' looks at the part '00000002'
                    ck_dataset['neutral_img'].append(img)
                else:
                    if(text == 1):
                        ck_dataset['anger_img'].append(img)
                    elif(text == 2):
                        ck_dataset['contempt_img'].append(img)
                    elif(text == 3):
                        ck_dataset['disgust_img'].append(img)
                    elif(text == 4):
                        ck_dataset['fear_img'].append(img)
                    elif(text == 5):
                        ck_dataset['happy_img'].append(img)
                    elif(text == 6):
                        ck_dataset['sadness_img'].append(img)
                    elif(text == 7):
                        ck_dataset['surprise_img'].append(img)
'''
  same code as in haar_apply.py but only break statement
  after the execution of elif becuase once we reache
  the direcory with images we will process all the
  image in directoy with function load_data()
'''
def trav_dir(dirpath):
    os.chdir(dirpath)
    dir_list = os.listdir()

    #travers current directory and if directoy found, call itself
    for x in dir_list:
      if(os.path.isdir(x)):
        trav_dir(x)
      #imghdr.what return mime type of the image
      elif(imghdr.what(x) in ['png']):
        load_data(dir_list)
        break

    #reached directory with no directory
    os.chdir('./..')

def pack_data():
    global ck_dataset
    biglist = [
        ck_dataset['neutral_img'], ck_dataset['anger_img'],
        ck_dataset['contempt_img'], ck_dataset['disgust_img'],
        ck_dataset['fear_img'], ck_dataset['happy_img'],
        ck_dataset['sadness_img'], ck_dataset['surprise_img']
    ]

    training_data = []
    validation_data = []
    test_data = []

    training_txt = []
    validation_txt = []
    test_txt = []

    for x,y in zip(biglist,range(0,8)):
        length = len(x)

        training_data.append(x[0:math.ceil(length*0.8)] ) #80 percent Image of each emotion for training
        training_txt.append(y*np.ones(shape=(len(x[0:math.ceil(length*0.8)]),1),dtype=np.int8)) #Generating Corresponding Label

        validation_data.append(x[math.ceil(length*0.8):math.floor(length*0.9)]) #10 percent
        validation_txt.append(y*np.ones(shape=(len(x[math.ceil(length*0.8):math.floor(length*0.9)]),1),dtype=np.int8))

        test_data.append(x[math.floor(length*0.9):length]) #10 percent
        test_txt.append(y*np.ones(shape=(len(x[math.floor(length*0.9):length]),1),dtype=np.int8))

    print(training_data)
    training_data = np.vstack( training_data) #np.vstack(list_of_array) converts the list_of_numpy_arrays into a single numpy array
    validation_data = np.vstack( validation_data)
    test_data = np.vstack( test_data)

    training_txt = np.vstack( training_txt)
    validation_txt = np.vstack( validation_txt)
    test_txt = np.vstack( test_txt)
    pickle.dump([[ training_data, training_txt],[ validation_data, validation_txt],[ test_data, test_txt]],open( outfile,'wb'))

def serialize_CK(dataset_path):
    trav_dir(dataset_path) #name of the directory
    pack_data()

serialize_CK(dataset_path)
