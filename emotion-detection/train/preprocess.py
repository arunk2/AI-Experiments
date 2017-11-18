import sys
import cv2
import dlib
from skimage import io
import numpy as np

cnt = 0
detector = dlib.get_frontal_face_detector()

argSize = len(sys.argv)
datadir = sys.argv[argSize-1]

print 'Outdirectory = '+datadir+' :'

for file in sys.argv[1:]:

    print("Processing file: {}".format(file))
    img = io.imread(file)

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    
    if(len(dets)!=0):
    	for i, d in enumerate(dets):
        l = d.left()
        t = d.top()
        r = d.right()
        b = d.bottom()   
        
    		pts = np.array([[l,t],[r,t],[r,b],[l,b]])
    		mask = np.zeros((img.shape[0],img.shape[1]))
    		cv2.fillConvexPoly(mask,pts,1)
        
    		mask = mask.astype(np.bool)
    		out = np.zeros_like(img)
    		out[mask] = img[mask]
        
    		(meanx, meany) = pts.mean(axis=0)
    		(cenx, ceny) = (img.shape[1]/2, img.shape[0]/2)
    		(meanx, meany, cenx, ceny) = np.floor([meanx, meany, cenx, ceny]).astype(np.int32)
    		(offsetx, offsety) = (-meanx + cenx, -meany + ceny)
    		(mx, my) = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    		ox = (mx - offsetx).astype(np.float32)
    		oy = (my - offsety).astype(np.float32)
        
    		out_translate = cv2.remap(out, ox, oy, cv2.INTER_LINEAR)
    		topleft = pts.min(axis=0) + [offsetx, offsety]
    		bottomright = pts.max(axis=0) + [offsetx, offsety]
    		cv2.rectangle(out_translate, tuple(topleft), tuple(bottomright), color=(255,0,0))
        
    		#cv2.imshow('Output Image', out_translate)

    		out = out_translate[topleft[1]:bottomright[1],topleft[0]:bottomright[0],:]
    		#cv2.imshow('image',out)
        
    		val = cv2.imwrite(datadir + str(cnt) + ".jpg" ,out)
    		cnt = cnt + 1 

