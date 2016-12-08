# Facial-Emotion-Recoginiton

#Overview
Using Extended Cohn-Kanade AU-Coded Facial Expression Database to classify basic human facial emotion expressions using ann

#Installation

The necessary dependencies are in the requirements.txt file so just run this before running the actual code to get them installed. May require some extra effort for OpenCV though.

``
pip3 install -r requirements.txt
``

#Usage

If you want a facial emotion classifier you can just download the Cohn-Kanade database and serialize the dataset according to your need and if you can you should also use another dataset with ck.

apply_haar\_cascade.py => To apply the haar cascade for face detection it will overwrite the directory with 100x100 pixels images of faces. If face couldn't be detected then print the filename to console.

pickle_dataset.py => Pickles the dataset containing 100*100 images of faces

cnn.py => Convolution neural Network to train the the classifier 

cohn-kanade-images.txt => contains some basic info about the dataset

#Credits
- Kanade, T., Cohn, J. F., & Tian, Y. (2000). Comprehensive database for facial expression analysis. Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG'00), Grenoble, France, 46-53.
- Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression. Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.
