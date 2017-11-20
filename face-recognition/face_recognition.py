import cv2
import os, sys
import pickle
import time
import numpy as np
np.set_printoptions(precision=2)
from align_dlib import AlignDlib
from torch_neural_net import TorchNeuralNet

def encodeFaces(imgPath):
    bgrImg = imgPath
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    faces = align.getAllFaceBoundingBoxes(rgbImg)
    
    if len(faces) == 0:    
        print "No Faces found !"
        raise Exception("Unable to find a face: {}".format(imgPath))

    codes = []
    for face in faces:
        alignedFace = align.align(
            96,
            rgbImg,
            face,
            landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))

        start = time.time()
        code = net.forward(alignedFace)
        #print("Neural network forward pass took {} seconds.".format(time.time() - start))
        codes.append((face.center().x, code))

    # Return sorted encoded images from left to right
    codes = sorted(codes, key=lambda x: x[0])
    return codes


def classify(img):
    with open(classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)

    reps = encodeFaces(img)

    for r in reps:
        rep = r[1].reshape(1, -1)
        bbx = r[0]
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]

        if confidence > .60:
            #print("Prediction took {} seconds.".format(time.time() - start))
            print("Predict {} @ x={} with {:.2f} confidence.".format(person, bbx,
                                                                         confidence))
        else:
            print ('Unknown person! @ x={}'.format(bbx))

    print(" ")
    return person, confidence


if __name__ == '__main__':
    align = AlignDlib("../models/dlib/shape_predictor_68_face_landmarks.dat")
    net = TorchNeuralNet("../models/nn4.small2.v1.t7", imgDim=96, cuda=False)
    classifier = 'LinearSvm'
    classifierModel = '../models/classifier.pkl'


    #video_path = '/home/dev/Work/python/ventuno/videos/SPORTS_sania_nehwal__3KQSG65R.mp4'
    #video_path = '/home/dev/Desktop/Rajini-Kamal/1_bbuzz_kabali_collection__BMEAPDZ0.mp4'
    video_path = '/home/dev/Desktop/Rajini-Kamal/1_ebuzz_kamal_chevaliar__9GBYQ2D5.mp4'
    #video_path = '/home/dev/Desktop/Rajini-Kamal/1_ebuzz_rajini_kamal_lyca__CLUNQP5A.mp4'
    #video_path = '/home/dev/Desktop/Rajini-Kamal/1_ebuzz_theri_kabali__17ZHDIUS.mp4'
    #video_path = '/home/dev/Desktop/Rajini-Kamal/1_ebuzz_vedhalam_teaser__W74FUJ8V.mp4'
    #video_path = '/home/dev/Desktop/Rajini-Kamal/Salman_Shahrukh.mp4'

    
    video_path = '/home/dev/Desktop/AIvideos/Anushka_Sharma_And_Virat_Kohli_Re_Unite_For_Ad_Shoot_Manyavar.mp4'
    video_path = '/home/dev/Desktop/AIvideos/After_coming_politics_kamal_will_not_act__N907G2ZO.mp4'
    video_path = '/home/dev/Desktop/AIvideos/Shah_Rukh_Khan_At_The_Launch_Of_Ted_Talk__9VYG2X6E.mp4'
    video_path = '/home/dev/Desktop/AIvideos/Anushka_on_working_with_the_Khans__2CWRNVBK.mp4'
    video_path = '/home/dev/Desktop/AIvideos/Srk__6KFMOJE9.mp4'
    video_path = '/home/dev/Desktop/AIvideos/Kajol_Rani_Mukherji_Alia_Bhatt_Sridevi_Karisma_In_Shahrukh_Khans_Film.mp4'
    # video_path = '/home/dev/Desktop/AIvideos/Dhanush_REACTION_on_Father_In_Law_Rajinikanth_Joining_BJP.mp4'
    # video_path = '/home/dev/Desktop/AIvideos/Murugadoss_Plan_Spider_audio_Release_fro__U3TGM521.mp4'
    # video_path = '/home/dev/Desktop/AIvideos/sg-23-09-2017-when-shah-rukh-khan-went-knocking-at-sanjay-dutts-door-for-help-jw-new.mp4'
    
    
    video_capture = cv2.VideoCapture(video_path)

    confidenceList = []
    frame_idx = 0
    while True:
        frame_idx = frame_idx + 1
        ret, frame = video_capture.read()

        if ret == False:
            break

        # Sample 1 image per second
        if frame_idx % 25 != 0:
            continue

        try:
            persons, confidences = classify(frame)
        except:
            # If there is no face detected, confidences matrix will be empty.
            # We can simply ignore it.
            pass

        cv2.imshow('', frame)
        cv2.waitKey(1)

        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
