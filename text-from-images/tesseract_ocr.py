from pytesseract import image_to_string 
import sys
from PIL import Image

imageFile = 'test.jpg'
print image_to_string(Image.open(imageFile),lang='eng')
