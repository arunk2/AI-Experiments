## Dependancies to be installed:

1) tensorflow
2) dlib
3) opencv
4) skimage

### Installation of Tensorflow:
There are many installation methods possible for tensorflow. 
The best and safe method is installation via virtual environment.

https://www.tensorflow.org/get_started/os_setup


### Installtion of dlib
```
git clone https://github.com/davisking/dlib.git  
python setup.py install
```

### Installtion of open cv
**Using Anaconda:** conda install opencv

**Using source:** http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/

### Installation of skimage
conda install skimage

## File documentation

| Files | Notes 
|----|-----
| preprocess.py | Extracts the face from the input image. 
| script.sh | Runs the tensorflow pretrained model and retrains it. [Refer](https://www.tensorflow.org/versions/master/how_tos/image_retraining/)
