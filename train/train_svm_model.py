import argparse
import cv2
import os, sys
import pickle

import pandas as pd
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

def train():
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(faceEmbeddingDir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/embedding.csv".format(faceEmbeddingDir)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)

    print("Training for {} classes.".format(nClasses))

    if classifier == 'LinearSvm':
        clf = SVC(C=1, kernel='linear', probability=True)
    else:
        raise Exception("Unknown Model !")

    clf.fit(embeddings, labelsNum)

    fName = "{}/classifier.pkl".format(faceEmbeddingDir)
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)


def partial_train():
    with open(args.classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)

    fname = "{}/labels.csv".format(faceEmbeddingDir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/embedding.csv".format(faceEmbeddingDir)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    labelsNum = le.transform(labels)

    print("Partial training......")
    clf.partial_fit(embeddings, labelsNum)
    print("Partial training completed !")

    fName = "{}/classifier_new.pkl".format(faceEmbeddingDir)
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)


if __name__ == '__main__':

    classifier = 'LinearSvm'
    faceEmbeddingDir = '/home/dev/face_recognition_nn_model/face-embeddings'

    # Take the train or partial_train @ command line with 
    # labels.csv & embedding.csv @ /home/dev/face_recognition_nn_model/face-embeddings
    if len(sys.argv) < 2:
        print 'Usage : python train_svm_model.py train/partial_train'
        sys.exit()
    
    if sys.argv[1] == 'train':
        train()
        sys.exit()
    elif sys.argv[1] == 'partial_train':
        partial_train()
        sys.exit()
    else:
        print 'Unknown Action !'
        sys.exit()
