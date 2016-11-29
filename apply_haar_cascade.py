import os
#for imghdr.what to find wheter a file is image
#['pbm','pgm','ppm','png','jpeg','tiff','bmp','webp']
import imghdr
import cv2

face_cascade = cv2.CascadeClassifier('c:/haarcascade_frontalface_default.xml')

def deal_with_image(imgpath):
	gray = cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
	faces = face_cascade.detectMultiScale(gray, 1.25, 5)
	if len(faces)==0:
			print(imgpath) # Change parameters of detectMultiScale or manually crop the image
	for (x,y,w,h) in faces:
		roi_gray = gray[y:y+h, x:x+w]
		roi_gray = cv2.resize(roi_gray, (100,100), interpolation = cv2.INTER_AREA)
		cv2.imwrite(imgpath , roi_gray )

def trav_dir(dirpath):
	os.chdir(dirpath)
	dir_list = os.listdir()

	#travers current directory and if directoy found call itself
	for x in dir_list:
		if(os.path.isdir(x)):
			trav_dir(x)
		#imghdr.what return mime type of the image
		elif(imghdr.what(x) in ['png']):
			deal_with_image(x)

	#reached directory with no directory
	os.chdir('./..')
						
#trav_dir('./mahakal/cohn-kanade-images')
