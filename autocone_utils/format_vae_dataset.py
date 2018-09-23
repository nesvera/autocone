#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from glob import glob
import getpass
import os
import shutil
import cv2

# Delete runs with less steps than min_seq
min_seq = 30

# Print iterations progress
def progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """

    total -= 1

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total: 
        print("")

if __name__ == "__main__":

    username = getpass.getuser()
    dataset_folder = '/home/'+ username + '/Documents/autocone_dataset/'

    vae_dataset_folder = '/home/'+ username + '/Documents/autocone_vae_dataset/'

    if not os.path.exists(dataset_folder):
        print("Directory doesnt exist! " + dataset_folder)
        exit(0)

    # get all run folders of the dataset
    runs_folders = glob(dataset_folder + "*/")
    runs_folders.sort(key=os.path.getmtime)
    runs_folder_len = len(runs_folders)

    for i, run_folder in enumerate(runs_folders):

        # get all images inside a run folder
        run_images = glob(run_folder + "*.jpg")

        # copy images to the vae folder
        for image in run_images:
            
            img = cv2.imread(image, 0)
            img = cv2.resize(img, None, fx=1/4., fy=1/4.)

            #print(img.shape)
            #cv2.imshow('image', img)
            #cv2.waitKey(0)

            img_name = os.path.basename(image)
            cv2.imwrite((vae_dataset_folder + img_name), img)


            #shutil.copy2(image, vae_dataset_folder)

        progress_bar(i, runs_folder_len, prefix='Progress', length=50)
