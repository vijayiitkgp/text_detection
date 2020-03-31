import pexpect
import os
from utils.classifier import Classifier
from config import *
from typing import Tuple
import wexpect

class DarknetClassifier(Classifier):
    

	def initialize(self):
		''' Initialize darknet. We do this for speed concerns.
		Input:
			thresh (float)   - specifies the threshold of detection
			data (string)    - name of the data file for darknet
			cfg (string)     - name of the configuration file
			weights (string) - name of the pre-trained weights
		Return:
		   proc (pexpect process), which we use to interact with the running darknet process '''

		# command = DARKNET_BINARY_LOCATION + " detector test " + DARKNET_DATA_FILE + " " + DARKNET_CFG_FILE \
		# 	+ " " + DARKNET_WEIGHTS + " -thresh " + str(DARKNET_THRESH) + " -ext_output -dont_show"

		command = DARKNET_BINARY_LOCATION + " detector test " + DARKNET_DATA_FILE + " " + DARKNET_CFG_FILE \
			+ " " + DARKNET_WEIGHTS
		print("command")
		if os.name == 'nt':
			# pexpect.popen_spawn.PopenSpawn
			self.proc = wexpect.spawn(command)
			print(command)
		else:
			self.proc = wexpect.spawn(command)
		# self.proc.expect('Enter Image Path:')
		# self.proc.expect('Enter Image Paths:')
		# self.proc.expect(pexpect.EOF, timeout=None)

	def classify_image(self, image):
		''' Classifies a given image. Simply provide the name (string) of the image, and the proc to do it on.
		Input:
			image (string)   - name of the saved image file
			self.proc (proc)      - Pexpect proc to interact with
		Return:
			Returns the output from darknet, which gives the location of each bounding box. '''
		print("image is here:", image)
		# child = wexpect.spawn('cmd')
		# child.expect('>')
		# child.sendline('ls')
		# child.expect('>')
		# print(child.before)
		# child.sendline('exit')
		child = wexpect.spawn('darknet.exe detector test data/obj.data yolov3-obj.cfg yolov3-obj_final.weights -thresh 0.25 -ext_output -dont_show pancards/dl.jpeg')
		# self.proc.expect('Enter Image Paths:')
		# self.proc.expect('.*')
		# self.proc.sendline(image)
		# self.proc.expect(pexpect.EOF, 'Enter Image Path:', timeout=90)
		# res = self.proc.before
		child.expect(wexpect.EOF)
		res = child.before
		print(res)
		# return res.decode('utf-8')
		return res
	def extract_info(self, line):
		''' Extracts the information from a single line that contains a label.
		Input: line (string), a line that already contains the label
		Output: area (Tuple of four ints), which gives the area of the bounding box.
		'''
		nameplate_info = line.split()
		#print(nameplate_info)
		nameplate_confidence = nameplate_info[1]
		try:
      
			nameplate_left_x = int(nameplate_info[3])
		except:
			nameplate_left_x = 0
		try:
			nameplate_top_y = int(nameplate_info[5])
		except:
			nameplate_top_y = 0
		try:
			nameplate_width = int(nameplate_info[7])
		except:
			nameplate_width = 2
		try:
			nameplate_height = int(nameplate_info[9][:-1])
		except:
			nameplate_height = 2
		# nameplate_width = int(nameplate_info[7])
		

		area = (nameplate_left_x, nameplate_top_y, (nameplate_left_x + nameplate_width), (nameplate_top_y + nameplate_height))

		return area
