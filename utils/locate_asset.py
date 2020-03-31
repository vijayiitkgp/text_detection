from PIL import Image
from PIL import ImageFilter
import utils.logger as logger
#from utils.rotate import rotate
from config import *
from typing import Tuple, List
import sys
i = 0
folder = "cropped"
def crop_image(image, area:Tuple) -> object:
	''' Uses PIL to crop an image, given its area.
	Input:
		image - PIL opened image
		Area - Coordinates in tuple (xmin, ymax, xmax, ymin) format '''
	img1 = Image.open(image)
	img = img1.crop(area)
	basewidth = 200
	try:
		wpercent = (basewidth/float(img.size[0]))
	except:
		wpercent = 	 (basewidth/float(5.00))
	# hsize = int((float(img.size[1])*float(wpercent)))
	hsize = 50
	# cropped_image = img.resize((basewidth,hsize), Image.ANTIALIAS)
	cropped_image = img
	global i
	cropped_image.save(folder + "\\"+ "r" + str(i) + ".jpg", "JPEG",dpi=(300,300))
	i += 1
	
	return cropped_image

def locate_asset(image, lines=""):
	
	cropped_images = []
	
	for line in lines:
		
			cropped_images.append((line, crop_image(image, line)))
		
	if cropped_images == []:
		logger.bad("No label found in image.")
	else:
		logger.good("Found " + str(len(cropped_images)) + " label(s) in image.")

	return cropped_images
