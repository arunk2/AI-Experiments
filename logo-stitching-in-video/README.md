Stitch Brand Logo Or Animation Inside Video Content
===================================================

## What is it?
A Computer vision-powered system that will analyse the entire video content and finds a suitable place for logo. It then put the processed logo in every frame of the video.

This work is inspired from the product [uru-video](http://uruvideo.com/)

## Algorithm:
1. Get the input logo size
2. Loop through all the frames and identify least varying place for logo placement (separate algorithm)
3. Loop through every frame
4. Identify changes in the identified area (using motion detection algorithm - separate algorithm) or ROI (region of interest)
5. Filter the logo which has alpha layer for the changes
6. Merge the processed logo and Original frame to create new frame
7. With the new set of frames create the new video


## Source code:

```
# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2


def display_overlay(threshold, img_logo, frame):
	# Load two images
	# threshold = cv2.imread('arun/threshold.png')
	# img_logo = cv2.imread('arun/75-logo-new.png', -1)
	
	# Now create a mask of logo and create its inverse mask also
	mask_inv = cv2.bitwise_not(threshold)

	channels = cv2.split(img_logo)
	channels[0] = cv2.bitwise_and(channels[0], channels[1], mask = mask_inv)
	channels[1] = cv2.bitwise_and(channels[1], channels[1], mask = mask_inv)
	channels[2] = cv2.bitwise_and(channels[2], channels[2], mask = mask_inv)
	channels[3] = cv2.bitwise_and(channels[3], channels[3], mask = mask_inv)

	res = cv2.merge(channels)

	# l_img = cv2.imread("arun/feed.png")
	l_img = frame
	s_img = res
	x_offset=350
	y_offset=450

	for c in range(0,3):
	    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], c] = s_img[:,:,c] * (s_img[:,:,3]/255.0) +  l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], c] * (1.0 - s_img[:,:,3]/255.0)


	cv2.imshow("Thresh", l_img)




camera = cv2.VideoCapture('arun/video_file.mp4')
 
# initialize the first frame in the video stream
firstFrame = None

# Logo to be overlayed
img_logo = cv2.imread('arun/75-logo-new.png', -1)

i = 0
# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()
	text = "Unoccupied"
 
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break
	
	if i < 3:
		i = i + 1
		continue
 
	# resize the frame, convert it to grayscale, and blur it
	# frame = imutils.resize(frame, width=500)

	crop_frame = frame[450:525, 350:650] # Crop from x, y, w, h -> 100, 200, 300, 400
	# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

	gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
 
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	# thresh = cv2.dilate(thresh, None, iterations=2)

	# show the frame and record if the user presses a key
	# cv2.imshow("Security Feed", frame)
	# cv2.imshow("Thresh", thresh)
	display_overlay(thresh, img_logo, frame)
	
	key = cv2.waitKey(1) & 0xFF

	cv2.imwrite('arun/threshold.png',thresh)
	cv2.imwrite('arun/feed.png',frame)
 
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

cv2.waitKey(0)

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()


```

### Reference:
http://uruvideo.com/
