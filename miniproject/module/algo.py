from __future__ import annotations
#Image proc imports
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import cv2
from cv2 import Mat
import os

#Other imports
from dataclasses import dataclass
import json

@dataclass
class HyperParametersBundle():
    """A single morph kernel is used throughought the project in order to reduce the
    number of kernel possibilities"""
    morph_kernel: tuple #Recommended: See cv2.getStructuringElement

    kmean_median_channel_morph: MorphOperation
    dist_transform_morh: MorphOperation
    binarized_dist_morph: MorphOperation

class ResultBundle():
    image_name: str 

    #Images
    original_image: Mat
    kmean_image: Mat
    suppressed_image: Mat
    distance_image: Mat
    distance_bin: Mat
    distance_bin_2: Mat
    marker_image: Mat
    watershed_image: Mat
    annotated_image: Mat
    kmean_bin_image: Mat

    #Quantified results
    count: int
    expected_count: list[int]
    diff: int

@dataclass
class MorphOperation():
    iterations: int
    type: int #cv2.MORPH_XXX (Open, close, dilate, erode)

class AlgorithmProcessor():
    labels: dict
    LABEL_FILENAME = "../dataset/json/data_labels.json"
    IMAGES_FILEPATH = "../dataset/png"

    #Algorithm constants (not every parameter is an hyper-parameter)
    ## Kmeans ##
    CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3 #Trying to find BG / general contours / bacteria
    ATTEMPTS = 10

    def __init__(self) -> None:
        try:
            with open(os.path.join(os.path.dirname(__file__) , self.LABEL_FILENAME) , "r") as f:
                self.labels = json.load(f)
        except Exception as e:
            print(f"Impossible to load label")
            raise IOError()

    def process(self, image_name: str, hyper_parameter_bundle: HyperParametersBundle) -> ResultBundle:
        output_bundle = ResultBundle()
        output_bundle.image_name = image_name
        image_label = self.labels["images"][image_name]

        #Load image
        image_filepath = os.path.join(os.path.dirname(__file__) , f"{self.IMAGES_FILEPATH}/{image_name}")
        image = cv2.cvtColor(cv2.imread(image_filepath), cv2.COLOR_BGR2RGB)
        output_bundle.original_image = image
        
        ### K-means ###
        #Apply K-means
        working_image = image.copy()
        image_as_array = np.float32(working_image.reshape((-1,3)))
        _, label, center = cv2.kmeans(image_as_array, self.K, None, self.CRITERIA, self.ATTEMPTS, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)

        #Find index of min/aver/max channels
        averages = [np.mean(c) for c in center]
        min_index = np.argmin(averages)
        max_index = np.argmax(averages)
        med_index = list(set((0,1,2)) - set((min_index, max_index)))[0]

        #Create 1D binary mask that is active only on medium channel
        binary_mask = np.vectorize(lambda x: 1 if x==med_index else 0)(label)
        #Transform binary mask into binary image that can be display and played with
        x,y,_ = image.shape
        binary_mask = binary_mask.reshape((x,y))
        mask_as_image = np.zeros_like(binary_mask, dtype=np.uint8)
        mask_as_image[binary_mask == 1] = 255
        #Apply morph on medium channel
        mask_morph = hyper_parameter_bundle.kmean_median_channel_morph
        if mask_morph.iterations > 0:
            mask_as_image = cv2.morphologyEx(mask_as_image, mask_morph.type, hyper_parameter_bundle.morph_kernel, iterations=mask_morph.iterations)
        output_bundle.kmean_image = mask_as_image

        #Compute supressed image by taking original one
        #And removing the modified median channel
        suppressed_image = image.copy()
        suppressed_image[mask_as_image != 0] = center[max_index]
        output_bundle.suppressed_image = suppressed_image

        #Binarize supressed image to obtain binarized image
        gray_image = cv2.cvtColor(suppressed_image, cv2.COLOR_RGB2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (5,5), 0)
        _, binarized_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        output_bundle.kmean_bin_image = binarized_image

        ### Distance transform ###

        #Apply dist transform and normalize
        dist_transform = cv2.distanceTransform(binarized_image, cv2.DIST_L2, 3, dstType=cv2.CV_8U)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        #Dist transform morph operation
        dist_morph = hyper_parameter_bundle.dist_transform_morh
        if dist_morph.iterations > 0:
            dist_transform = cv2.morphologyEx(dist_transform, dist_morph.type, hyper_parameter_bundle.morph_kernel, iterations=dist_morph.iterations)
        output_bundle.distance_image = dist_transform
    
        # Threshold dist transform 
        _, dist = cv2.threshold(dist_transform, 0.4, 1.0, cv2.THRESH_BINARY)
        output_bundle.distance_bin = dist
        # apply yet anoter morph
        binarized_morph = hyper_parameter_bundle.binarized_dist_morph
        if binarized_morph.iterations > 0:
            dist = cv2.morphologyEx(dist, binarized_morph.type, hyper_parameter_bundle.morph_kernel, iterations=binarized_morph.iterations)
        output_bundle.distance_bin_2 = dist

        # Create the marker images
        dist_8u = dist.astype('uint8')
        contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        markers = np.zeros(dist.shape, dtype=np.int32)
        # Draw the markers
        for i in range(len(contours)):
            cv2.drawContours(markers, contours, i, (i+1), -1)

        output_bundle.marker_image = markers

        #Count - Watershed is for the bacteria's shape, counting is already available now after distance transform
        bacteria_count = len(contours)
        label_bacteria_count: list[int] = image_label["bacteryCount"]
        output_bundle.count = bacteria_count
        output_bundle.expected_count = label_bacteria_count

        if bacteria_count in label_bacteria_count:
            diff = 0
        elif bacteria_count > max(label_bacteria_count):
            diff = bacteria_count-max(label_bacteria_count)
        else:
            diff = min(label_bacteria_count)-bacteria_count
        output_bundle.diff = diff

        ### Optional- WATERSHED ###

        # Perform the watershed algorithm
        watershed_result = cv2.watershed(suppressed_image, markers)

        # Generate random colors
        colors = []
        for i in range(len(contours)):
            colors.append((randint(0,256), randint(0,256), randint(0,256)))

        # Create the result image
        watershed_image = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
        # Fill labeled objects with random colors
        for i in range(markers.shape[0]):
            for j in range(markers.shape[1]):
                index = markers[i,j]
                if index > 0 and index <= len(contours):
                    watershed_image[i,j,:] = colors[index-1]
        
        output_bundle.watershed_image = watershed_image

        # Show contour on original image image
        result_image = image.copy()
        result_image[watershed_result == -1] = [255,0,0]
        output_bundle.annotated_image = result_image

        return output_bundle