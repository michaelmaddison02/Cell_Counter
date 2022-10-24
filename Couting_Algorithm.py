from cs1lib import *
import math
import os

import cv2
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import time
import concurrent.futures
import image_slicer
from image_slicer import join
import shutil

from scipy.ndimage import label

#https://stackoverflow.com/questions/58751101/count-number-of-cells-in-the-image
from scipy import ndimage
from skimage.feature import peak_local_max
#from skimage.morphology import watershed

import imutils
import argparse
from skimage.segmentation import watershed
import skimage.color
import skimage.io


class counting_algorithm:
    def __init__(self):
        self.gray_thresh = 1
        self.cell_list = []
        self.min_cell_size = 20

        self.live_dead = None           #image of live or dead cells
        self.cell_type = None           #"contr", "rod", "hrpc", "cone"
        self.concentration = None       #concentration of h202
        self.primary_secondary = None   #primary or secondary image of slide

        self.image_dict = {}  # main dictionary of images

    #parameter: integers
    #return: boolean
    def gray_match(self, c1, c2):

        # if within threshold
        if abs(c1 - c2) <= self.gray_thresh:
            return True
        else:
            return False

    #helper function
    # return: numpy array of (h) x (w)
    def make_white_image(self, height, width):
        img = np.zeros([height, width, 1], dtype=int)
        # img.fill(0)  # numpy array!
        # im = Image.fromarray(img)  # convert numpy array to image
        # im.save('whh.jpg')

        return img

    #helper function
    def get_baseline_threshold(self, gray_image):
        height = gray_image.shape[0]
        width = gray_image.shape[1]

        avg_luminance = 0
        pixels_counted = 0

        for x in range(width):
            for y in range(height):

                #if the pixel is not black (background)
                if not self.gray_match(gray_image[y, x], 0):
                    avg_luminance += gray_image[y, x]
                    pixels_counted += 1

        if pixels_counted > 0:
            avg_luminance = avg_luminance / pixels_counted

        #print("threshold: ", avg_luminance)
        return avg_luminance

    def connected_component_analysis(self, image, main_dict):
        # creates grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #get baseline threshold
        threshold = self.get_baseline_threshold(gray)

        # get threshold
        ret, thresh = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)
        #self.image_dict[main_dict]['thresh'] = thresh

        #ostu thresholding
        #otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # noise removal
        kernel = np.ones((6, 6), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        #erode image
        erode = cv2.erode(sure_fg, kernel, iterations=1)

        self.new_count_cells(thresh)
        self.new_count_cells(erode)

    def breakdown_filename(self, filename):
        breakdown = ''
        cell_types = ["contr", "rod", "hrpc", "cone"]   #detect cell type
        for type in cell_types:
            if type in filename:
                breakdown += type
                self.cell_type = type
                break

        live_dead = ["ch00", "ch01"]    #detect live or dead image
        if live_dead[0] in filename:
            breakdown += " live"
            self.live_dead = "live"
        elif live_dead[1] in filename:
            breakdown += " dead"
            self.live_dead = "dead"
        else:
            breakdown += " combined"
            self.live_dead = "combined"

        components = filename.split("_ch")      #detect concentration
        concentrations = ["1", ".5", ".25", "0"]
        for concentration in concentrations:
            if concentration in components[0]:
                breakdown += " " + concentration
                self.concentration = concentration
                break

        if "hrpc" in filename:                  #detect primary or secondary, account for 'p' in hrpc
            if filename.count('p') == 1:
                breakdown += " primary"
                self.primary_secondary = "primary"
            else:
                breakdown += " secondary"
                self.primary_secondary = "secondary"
        else:
            if filename.count('p') == 0:
                breakdown += " primary"
                self.primary_secondary = "primary"
            else:
                breakdown += " secondary"
                self.primary_secondary = "secondary"
        return breakdown

    #helper function
    #purpose, reads image folder, adds all images as keys to a dictionary, value is a sub-dictionary
    def catalogue_images(self, rootdir):

        for subdir, dirs, files in os.walk(rootdir):  #walk through all filenames
            for filename in files:      #add filename as key to dictionary, value is subdict
                if filename not in self.image_dict: #only add filename if not in dictionary
                    image_subdict = {}
                    self.image_dict[filename] = image_subdict

    #main function
    #fully process all images. displays all processed images
    def handle_image_processing(self, folder_name):
        rootdir = "images/" + folder_name + "/"

        #creates dictionary of the images
        self.catalogue_images(rootdir)

        #index through image dictionary
        for filename in self.image_dict.keys():
            original = cv2.imread(rootdir + filename)

            #add title and original image to subdict
            self.image_dict[filename]['title'] = self.breakdown_filename(filename)
            self.image_dict[filename]['original'] = original

            #apply watershed to image
            self.watershed_v1(original, filename)

            print(self.image_dict[filename]['title'], ", cell count: ", self.image_dict[filename]['cell_count'], "     " + filename)

        self.display_all_images()

    def display_all_images(self):
        # https://www.codegrepper.com/code-examples/python/how+to+show+multiple+images+in+python
        columns = 3
        rows = len(self.image_dict.keys())
        image_index = 0

        fig = plt.gcf()
        fig.set_size_inches(columns * 4, rows * 4)

        # display original, thresh, watershed for each image
        for filename in self.image_dict.keys():
            self.subplots(filename, columns, rows, image_index)
            image_index += 3
        plt.show()

    def display_image(self, filename):
        if filename in self.image_dict.keys():
            columns = 3
            rows = 1
            image_index = 0

            self.subplots(filename, columns, rows, image_index)
            plt.show()
        else:
            print("image not found")

    def subplots(self, filename, columns, rows, image_index):
        # create figure
        fig = plt.gcf()
        fig.set_size_inches(columns * 6, rows * 6)

        # Add a subplot, show original
        image_index += 1
        fig.add_subplot(rows, columns, image_index)
        plt.imshow(self.image_dict[filename]['gray'], cmap='gray')
        plt.title("gray")

        # show thresh
        image_index += 1
        fig.add_subplot(rows, columns, image_index)
        plt.imshow(self.image_dict[filename]['thresh'], cmap='gray')
        plt.title("thresh")

        # show watershed
        image_index += 1
        fig.add_subplot(rows, columns, image_index)
        plt.imshow(self.image_dict[filename]['watershed'])
        plt.title("watershed")

    #saves thresh, watershed, cell_count to image_dict
    def watershed_v1(self, image, filename):
        #image in grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = self.get_baseline_threshold(gray)
        self.image_dict[filename]['gray'] = gray

        #thresholded image using baseline thresh.
        thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        self.image_dict[filename]['thresh'] = thresh

        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)

        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)

        #print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

        #information to identify cells
        minimum_area = 0
        average_cell_area = 650
        connected_cell_area = 1000
        cells = 0

        # loop over the unique labels returned by the Watershed algorithm
        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background' so simply ignore it
            if label == 0:
                continue

            # otherwise, allocate memory for the label region and draw it on the mask
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255

            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

            area = cv2.contourArea(c)
            if area > minimum_area:
                cv2.drawContours(image, [c], -1, (36, 255, 12), 2)
                # if area > connected_cell_area:
                # 	cells += math.ceil(area / average_cell_area)
                # else:
                cells += 1
        self.image_dict[filename]['watershed'] = image
        self.image_dict[filename]['cell_count'] = cells
