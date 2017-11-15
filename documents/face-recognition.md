## Face Recognition

Face Recongnition of a preprocessed image has 2 steps.
1. Face encoding - Converting the RGB-Pixels to numeric representation.
2. Name(or Classify) the encoded image into class of known images

> NOTE: For both encoding and classification, we need have to trained models.

### Training:

For both encoding and classification to work, we need labelled training data. i.e. set of preprocessed face images of known persons.

For **Encoding process**, we will train a Deep Convolutional Neural Network(DNN), for correctly encoding human faces.
![How DNN works?](../images/DNN1.png?raw=true "DNN for object detection")
![How DNN works?](../images/DNN2.png?raw=true "DNN for face detection")

For **Naming process**, we will train a SVM classifier, to correctly classify a encoded image into label (Names of persons)
![How SVM works?](../images/SVM.png?raw=true "SVM Classification")


### Image Encoding:
The simple approch for a image recognition, would be to compare a given unknown face with all the known pictures. When we find face that looks very similar to the unknown face, we found our person. But this approach won't scale for millons or billions of known pictures.
The next approach would be to obtain features from face and represent it in some way so that machine could use it for comparison (rather than pixel-by-pixel matching).

### Problem:
Identifying what features to extract from each face to build our known face database? Ear size? Nose length? Eye color? Something else? 

### Solution:
Researchers in DNN have discovered that the most accurate approach is to let the computer figure out the measurements to collect itself. Deep learning does a better job than humans at figuring out which parts of a face are important to measure. Let us train a Deep Convolutional Neural Network to generate 128 measurements for each face. 

### Training Flow:
For every image in the training set, we take 2 other face images
1. Load a training face image of a known person
2. Load another picture of the same known person
3. Load a picture of a different person

Then the algorithm looks at the measurements it is currently generating for each of those three images. It then tweaks the neural network slightly so that it makes sure the measurements it generates for #1 and #2 are slightly closer while making sure the measurements for #2 and #3 are slightly further apart.

After repeating this step millions of times for millions of images of thousands of different people, the neural network learns to reliably generate 128 measurements for each person. After this successful training, any set of pictures from the same person should give roughly the same measurements.

> Note 1: We used google's FaceNet approach in training our system. (http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)

> Note 2: Our code is highly inspired by openface implementation. (http://cmusatyalab.github.io/openface/)

--------------------------

## Encoding to Name of person (of person recognized):
**Problem:** Finding a person in our database of known people, who has the closest measurements to the given face image.
**Solution:** We use SVM classifier for this task. All we need to do is train a classifier that can take in the 128 measurement of known person as data and their name as label. 

Then for any 128 measurement, this classifier will predict a label (name of a person) with some confidence.

## Reference: 
- We used google's FaceNet approach in training our system. (http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)
- Our code is highly inspired by openface implementation. (http://cmusatyalab.github.io/openface/)
