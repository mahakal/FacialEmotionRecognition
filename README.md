# Facial-Emotion-Recoginiton

## Overview
Using Extended Cohn-Kanade AU-Coded Facial Expression Database to classify basic human facial emotion expressions using ann.

## Installation

The necessary dependencies are in the requirements.txt file so just run this before running the actual code to get them installed. May require some extra effort for OpenCV though.

``
pip3 install -r requirements.txt
``

## Usage

If you want a facial emotion classifier you can just download the Cohn-Kanade dataset and serialize the dataset according to your need and if you can you should also use another dataset with ck for better results.

pickle_dataset.py => Script for processing and Serializing dataset.
  python pickle_dataset.py -h

cnn.py => Convolution Neural Network to train the the classifier.
  python cnn.py -h

cohn-kanade-images.txt => Contains some basic info about the dataset.

## Credits
- Kanade, T., Cohn, J. F., & Tian, Y. (2000). Comprehensive database for facial expression analysis. Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG'00), Grenoble, France, 46-53.
- Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression. Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.
