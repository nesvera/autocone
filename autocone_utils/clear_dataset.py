#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Pass through all folders of the dataset set (wich run is stored inside a folder). 
    - Delete folders(runs) with less than 1 second of simulation (30 frames)
'''

from __future__ import print_function

from glob import glob
import getpass
import os
import shutil

# Delete initial frames (car respawning)
initial_frames = 5

# Delete runs with less steps than min_seq
min_seq = 5

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
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total: 
        print("")

if __name__ == "__main__":

    username = getpass.getuser()
    dataset_folder = '/home/'+ username + '/Documents/autocone_dataset/'

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
        run_len = len(run_images)

        # if bigger than min, just delete initial frames
        if run_len >= (initial_frames+min_seq):
            pass

        # else, delete the folder 
        else:
            #print("Deleting " + run_folder)
            shutil.rmtree(run_folder)
        
        progress_bar(i, runs_folder_len, prefix='Progress', length=50)


