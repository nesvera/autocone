#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2 as cv
import pandas as pd

from matplotlib import pyplot as plt


class Preprocess:

	def __init__(self, path, dataname):
		self.path = path + '/' + dataname
		self.dataname = dataname
		self.files_list, self.all_images = self.list_images()
		self.n = len(self.files_list)
		self.data_list = self.text_data_list()

	def list_images(self):
		filelist = [file for file in sorted(os.listdir(self.path)) if file.endswith('.jpg')]
		all_imgs = [cv.imread(self.path+file, 0) for file in filelist]
		resized = self.resize(all_imgs)
		return filelist, resized

	def resize(self, imgs):
		resized = [cv.resize(img, None,fx=0.3, fy=0.3, interpolation = cv.INTER_CUBIC) for img in imgs]
		return resized

	def text_data_list(self):
		filename = self.path + '/' + self.dataname + '.txt'
		data = pd.read_csv(filename, sep=";", header=None)
		data.columns = ["Name", "Speed", "Steering", "Collision", "*"]
		return data
		# print(data.columns)

	def show_image_and_data(self):
		rand = np.random.randint(0, self.n) # select random image from n

		data = self.data_list.iloc[rand]
		print('Filename: {0}, Speed: {1}, Steering: {2}, Collision: {3} ').format(data[0], data[1], data[2], data[3])
		# steering: -1 = right, 1 = left

		plt.imshow(self.all_images[rand])
		plt.show()

	def return_data(self):
		return self.all_images, self.data_list, self.n

