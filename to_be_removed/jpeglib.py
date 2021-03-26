import numpy as np
import subprocess
from PIL import Image

djpeg_location = '/Users/lindamegner/MATS/retrieval/jpeg-6b/test/bin/djpeg'

def read12bit_jpegfile(filename):
	batcmd = djpeg_location + ' -grayscale -pnm ' + filename #call jpeg decompression executable
	imagedata = subprocess.check_output(batcmd,shell=True) #load imagedata including header
	
	#read image as manually since 2 byte ppm format not supported by standard python library
	
	imagedata = imagedata.split(b"\n", 3)  #plit into magicnumber, shape, maxval and data
	
	imsize = imagedata[1].split() #size of image in height/width
	imsize = [int(i) for i in imsize]
	imsize.reverse() #flip size to get width/heigth
	maxval = int(imagedata[2])
	
	im = np.frombuffer(imagedata[3], dtype=np.uint16) #read image data
	
	im = im.reshape(imsize) #reshape image
	
	return im
	
def read16bit_jpegfile(filename):
	im_object = Image.open('test.jpg')
	im = np.asarray(im_object)
	return im
