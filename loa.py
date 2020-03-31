#!/bin/bash
from __future__ import print_function
from config import *
from utils.darknet_classify_image import *
from utils.tesseract_ocr import *
from ctpn.demo_pb import classify
import utils.logger as logger
from utils import locate_asset
import sys
from PIL import Image
import time
import os
import re
from operator import itemgetter
PYTHON_VERSION = sys.version_info[0]
OS_VERSION = os.name
import pandas as pd
from de_skew_image import deskew


import glob
import os
import shutil
import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

sys.path.append(os.getcwd())
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import _get_blobs
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.rpn_msr.proposal_layer_tf import proposal_layer



import cv2
import numpy as np

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def box_extraction(img_for_box_extraction_path, cropped_dir_path):

    img = cv2.imread(img_for_box_extraction_path, cv2.IMREAD_COLOR) 
     # Read the image
    img = deskew(img)
    cv2.imwrite(img_for_box_extraction_path,img)
    
    (thresh, img_bin) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    # print(img_bin)
    cv2.imwrite("Image_bin.jpg",img_bin)
    img_bin = 255-img_bin  # Invert the image

    cv2.imwrite("Image_bin_inverted.jpg",img_bin)
   
    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40
    print("printing the kernel length", kernel_length)
     
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    cv2.imwrite("verticle_lines.jpg",verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    cv2.imwrite("img_final_bin.jpg",img_final_bin)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    # print(contours, hierarchy)
    print(hierarchy)
    # [Next, Previous, First_Child, Parent]
    
#     CV_RETR_EXTERNAL retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.
# CV_RETR_LIST retrieves all of the contours without establishing any hierarchical relationships.
# CV_RETR_CCOMP retrieves all of the contours and organizes them into a two-level hierarchy. At the top level, there are external boundaries of the components. At the second level, there are boundaries of the holes. If there is another contour inside a hole of a connected component, it is still put at the top level.
# CV_RETR_TREE
    
#     CV_CHAIN_APPROX_NONE stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.
# CV_CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.
# CV_CHAIN_APPROX_TC89_L1,CV_CHAIN_APPROX_TC89_KCOS applies one of the flavors of the Teh-Chin chain approximation algorithm. See [TehChin89] for details.
    # contours, hierarchy = cv2.findContours(
        # img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    idx = -1
    names = ["dob", "name" , "sign"]
    for c, he in zip(contours, hierarchy[0]):
        
        if(he[1]>1):
            
            continue
        # Returns the location and width,height for every contour
        # print(he[1])
        x, y, w, h = cv2.boundingRect(c)

        # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        if (idx<2 and w > 80 and h > 20 and w > 3*h):
            idx += 1
            new_img = img[y:y+h, x:x+w]
            # print(he)
            print(idx)
            cv2.imwrite(cropped_dir_path+str(names[idx]) + '.png', new_img)



class PanOCR():
    ''' Finds and determines if given image contains required text and where it is. '''

    def init_vars(self):
        try:
            
            self.TESSERACT = TESSERACT
            

            return 0
        except:
            return -1

    def init_classifier(self):
        ''' Initializes the classifier '''
        try:
            if self.DARKNET:
            # Get a child process for speed considerations
                logger.good("Initializing Darknet")
                self.classifier = DarknetClassifier()
            
            if self.classifier == None or self.classifier == -1:
                
                print("here")
                return -1
            return 0
        except:
            return -1

    def init_ocr(self):
        ''' Initializes the OCR engine '''
        try:
            if self.TESSERACT:
                logger.good("Initializing Tesseract")
                self.OCR = TesseractOCR()
            
            if self.OCR == None or self.OCR == -1:
                return -1
            return 0
        except:
            return -1

    def init_tabComplete(self):
        ''' Initializes the tab completer '''
        try:
            if OS_VERSION == "posix":
                global tabCompleter
                global readline
                from utils.PythonCompleter import tabCompleter
                import readline
                comp = tabCompleter()
                # we want to treat '/' as part of a word, so override the delimiters
                readline.set_completer_delims(' \t\n;')
                readline.parse_and_bind("tab: complete")
                readline.set_completer(comp.pathCompleter)
                if not comp:
                    return -1
            return 0
        except:
            return -1

    def prompt_input(self):
		
		
        filename = str(input(" Specify File >>> "))
		

	# from utils.locate_asset import locate_asset

    def initialize(self):
        if self.init_vars() != 0:
            logger.fatal("Init vars")
        if self.init_tabComplete() != 0:
            logger.fatal("Init tabcomplete")
        # if self.init_classifier() != 0:
        # 	logger.fatal("Init Classifier")
        if self.init_ocr() != 0:
            logger.fatal("Init OCR")
	

    def find_and_classify(self, filename):
     
        start = time.time()


        #------------------------------Classify Image----------------------------------------#

            
        # logger.good("Classifying Image")


        coords= classify(filename)
        # print(len(coords))
        if(len(coords))>0:
            print("TEXT FOUND")
        else:
            print("NO TEXT FOUND")
            
            
        

		# # ----------------------------Crop Image-------------------------------------------#
        # logger.good("Finding required text")

        # cropped_images = locate_asset.locate_asset(filename, lines=coords)


        # time2 = time.time()


		
		# #----------------------------Perform OCR-------------------------------------------#
		
        # ocr_results = None

        # if cropped_images == []:
        #     logger.bad("No text found!")
        #     return None 	 
        # else:
        #     logger.good("Performing OCR")
        #     ocr_results = self.OCR.ocr(cropped_images)
        #     print(ocr_results)
        # 	k=[]
		# 	v=[]
			
			
		# 	fil=filename+'-ocr'
		# 	#with open(fil, 'w+') as f:
		# 	for i in range(len(ocr_results)):
					
		# 					v.append(ocr_results[i][1])
		# 					k.append(inf[i][0][:-1])
							
		# 	#k.insert(0,'Filename')
		# 	#v.insert(0,filename)
		# 	t=dict(zip(k, v))
			

		
		# time3 = time.time()
		# print("OCR Time: " + str(time3-time2))

		# end = time.time()
		# logger.good("Elapsed: " + str(end-start))
		# print(t)
		# return t
		
		
			
		#----------------------------------------------------------------#

    def __init__(self):
        ''' Run PanOCR '''
        self.initialize()

if __name__ == "__main__":
    
    # box_extraction("LOA_signed.jpg", "./Cropped/")
    
    extracter = PanOCR()
    tim = time.time()
	


    data=[]
  
  

    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    cfg_from_file('ctpn/text.yml')

    # init session
    loa_names = glob.glob(os.path.join(cfg.DATA_DIR, 'loa', '*.png')) + glob.glob(os.path.join(cfg.DATA_DIR, 'loa', '*.jpg'))
    
    saved_dir = os.path.join(cfg.DATA_DIR, 'demo\\')
    # print(saved_dir)
    for loa in loa_names:
        box_extraction(loa, saved_dir)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    for filename in im_names:

        # for filename in os.listdir('pancards'):
        # 	print(filename)
        # 	filename='pancards/'+filename
            print(filename)
            extracter.find_and_classify(filename)
            # result=extracter.find_and_classify(filename)
            #print(df1)
            #df=df.append(df1)
            # if result==None:
            # 	continue
            # else:
            # 	data.append(result)
        
            # df=pd.DataFrame(data)
            # #print(df)
            # df.to_csv (r'output/ocr_result_pan.csv', index = None, header=True,sep='\t')
            # en = time.time()
            # print('TOTAL TIME TAKEN',str(en-tim))
