import os
import imghdr
import cv2
import pickle
import math
import numpy as np

class PickleData(object):

  def __init__(self):
    #using list because append is costly on numpy array/ need to hardcode the array size
    self.neutral_img = []
    self.anger_img = []
    self.contempt_img = [] 
    self.disgust_img = []
    self.fear_img = []
    self.happy_img = []
    self.sadness_img = []
    self.surprise_img = []
    
  '''
  traverse directory tree if a directory found with
  image then find if .txt file exist if exist and 
  start collecting data
  '''
  def deal_with_data(self,list):
    if any(".txt" in s for s in list): #check whether directory have emotion label
    
      threshold = math.floor( (len(list)-1)*0.3) #how many pictures be considered neutral in directory
      
      for x in list: #find emotion label and read it 
        if ".txt" in x:
          text_file = open(x,'rb')
          text = int(float(text_file.readline()))
          text_file.close()
          break
          
      for x in list:
        if imghdr.what(x) in ['png']: #Makes sure file is .png image
          img = cv2.imread(x,cv2.IMREAD_GRAYSCALE) 
          img = img.flatten() #changes 100x100 to 10000 in a row major fashion
          img = img/25500 # normalization of data [for sigmoid neurons(0-1)]
          img = img.astype(np.float32) 
          if int(x[10:-4]) < threshold: #Image name are of type 'S005_001_00000002.png' looks at the part '00000002'
            self.neutral_img.append(img)
          else:
            if(text == 1):
              self.anger_img.append(img)
            elif(text == 2):
              self.contempt_img.append(img)
            elif(text == 3):
              self.disgust_img.append(img)
            elif(text == 4):
              self.fear_img.append(img)
            elif(text == 5):
              self.happy_img.append(img)
            elif(text == 6):
              self.sadness_img.append(img)
            elif(text == 7):
              self.surprise_img.append(img)
  '''
  same code as in haar_apply.py but only break statement
  after the execution of elif becuase once we reache 
  the direcory with images we will process all the 
  image in directoy with function deal_with_data()
  '''
  def trav_dir(self,dirpath):
    os.chdir(dirpath)
    dir_list = os.listdir()

    #travers current directory and if directoy found call itself
    for x in dir_list:
      if(os.path.isdir(x)):
        self.trav_dir(x)
      #imghdr.what return mime type of the image
      elif(imghdr.what(x) in ['png']):
        self.deal_with_data(dir_list)
        break

    #reached directory with no directory
    os.chdir('./..')

  def pack_data(self):
  
    biglist = [self.neutral_img, self.anger_img, self.contempt_img, self.disgust_img, self.fear_img, self.happy_img, self.sadness_img, self.surprise_img]
    
    self.training_data = []
    self.validation_data = []
    self.test_data = []
    
    self.training_txt = []
    self.validation_txt = []
    self.test_txt = []
    
    for x,y in zip(biglist,range(0,8)):
      length = len(x)

      self.training_data.append(x[0:math.ceil(length*0.8)] ) #80 percent Image of each emotion for training
      self.training_txt.append(y*np.ones(shape=(len(x[0:math.ceil(length*0.8)]),1),dtype=np.int8)) #Generating Corresponding Label

      self.validation_data.append(x[math.ceil(length*0.8):math.floor(length*0.9)]) #10 percent
      self.validation_txt.append(y*np.ones(shape=(len(x[math.ceil(length*0.8):math.floor(length*0.9)]),1),dtype=np.int8))

      self.test_data.append(x[math.floor(length*0.9):length]) #10 percent
      self.test_txt.append(y*np.ones(shape=(len(x[math.floor(length*0.9):length]),1),dtype=np.int8))
    
    self.training_data = np.vstack(self.training_data) #np.vstack(list_of_array) converts the list_of_numpy_arrays into a single numpy array
    self.validation_data = np.vstack(self.validation_data)
    self.test_data = np.vstack(self.test_data)
    
    self.training_txt = np.vstack(self.training_txt)
    self.validation_txt = np.vstack(self.validation_txt)
    self.test_txt = np.vstack(self.test_txt)

  def pickle_data(self):
    pickle.dump(((self.training_data,self.training_txt),(self.validation_data,self.validation_txt),(self.test_data,self.test_txt)),open('cohn_dataset.p','wb'))

  def start_pickling(self):
    self.trav_dir('./mahakal/cohn-kanade-images') #name of the directory
    self.pack_data()
    self.pickle_data()

#import p
#p1 = p.PickleData()
#p1.start_pickling()
