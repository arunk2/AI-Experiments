## Why pre-processing?
Pre-processing of face images is really important before going for a match.
This will remove any additional information might deviate the classifier and also avoid the issue with different poses of the same person.


## How we do this?
For this we will identify variuos features of the faces like eyes, nose, lips, etc. in the given image and adjust the image to put them in a straight pose. This is a 2 step process. 
1. Indentify face features (eyes, nose, lips, chin, etc.) or also known as face landmarks.
2. Transform face image, so that all the features are straight and front facing.


### Step 1: 
The basic idea is we will come up with 68 specific points (called landmarks) that exist on every human face — the top of the chin, the outside edge of each eye, the inner edge of each eyebrow, etc. Then we will train a machine learning algorithm to be able to find these 68 specific points on any face:

### Step 2: 
Now that we know were the eyes and mouth are, we’ll simply rotate, scale and shear the image so that the eyes and mouth are centered as best as possible (only affine transformations). 


## Sample code:

```
import sys
import dlib
import cv2
from align_dlib import AlignDlib

def hog_detection(image):
	# HOG face detector - dlib class
	face_detector = dlib.get_frontal_face_detector()

	detected_faces = face_detector(image, 1)
	# Detect faces, param 3 - threshold
	detected_faces, score, idx = face_detector.run(image, 1, -.5)
	return detected_faces


def align_face(image, detected_faces, face_width):
	# download the model http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
	predictor_model = "../models/dlib/shape_predictor_68_face_landmarks.dat"
	face_pose_predictor = dlib.shape_predictor(predictor_model)

	# Loop through each face
	for i, face_rect in enumerate(detected_faces):
		# Get the the face's pose - so that we can align with
		points = face_pose_predictor(image, face_rect)
		landmarks = list(map(lambda p: (p.x, p.y), points.parts()))

		# Align the face by some affine transformation.
		alignedFace = align(face_width, image, face_rect, landmarks, landmarkIndices=OUTER_EYES_AND_NOSE)

		# Save the aligned image to a file
		cv2.imwrite("a_{}.jpg".format(i), alignedFace)


def align(imgDim, rgbImg, bb=None, landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP):
    #Convert the landmark points to float-point array
    npLandmarks = np.float32(landmarks)
    npLandmarkIndices = np.array(landmarkIndices)
    #Apply Affine transformation to match the Mean FACE Image
    H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],
                               imgDim * MINMAX_TEMPLATE[npLandmarkIndices])
    #Resize  the FACE Image
    thumbnail = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))

    return thumbnail


if __name__ == '__main__':
	# Take the image file name from the command line
	if len(sys.argv) < 2:
		print 'Usage : python align_faces.py <<IMAGE_FILE_LOCATION>>'
		sys.exit()
		
	file_name = sys.argv[1]
	# Load the image
	image = cv2.imread(file_name)
	# Detect faces
	faces = hog_detection(image)
	# Proprocess - align the image
	if len(faces) > 0:
		face_width = 100
		align_face(image, faces, face_width)

```

## Output:
For any given face now, irrespective if its pose and position - center the eyes and mouth are in roughly the same position in the image. This will make our next step a lot more accurate.


## Reference:
- https://github.com/davisking/dlib
- https://github.com/cmusatyalab/openface
