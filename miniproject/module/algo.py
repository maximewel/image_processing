from __future__ import annotations
#Image proc imports
from random import randint
import numpy as np
import cv2
from cv2 import Mat
import os
from typing import Tuple, List
import json
import time

from data_classes import MorphOperation, HyperParametersBundle, ResultBundle

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
            raise IOError(str(e))

    # Operation from Gregoire Rebstein at, see: https://github.com/Drawfan/image-processing/blob/main/project/counting_bacteria.ipynb
    def reconstruct(self, image, markers):
        """Reconstruct an image according to the list of markers - rebuild markers if they don't existe anymore"""
        for i in range(1, markers.max() + 1):
            if len(set(image[markers==i])) == 1:
                image[markers == i] = 255

    def apply_morphology(self, image: Mat, kernel: Tuple, morph_operation: MorphOperation, repair_lost: bool) -> Mat:
        """Apply the given morphology operation on the image in place.
        
        Args
        -----


        Returns
        -----

        """

        iterations = morph_operation.iterations


        if (isinstance(iterations, int) and iterations <= 0):
            return image

        if isinstance(iterations, int):
            iterations = [iterations]

        for iteration in iterations:
            if repair_lost:
                _, markers = cv2.connectedComponents(image)
                image = cv2.morphologyEx(image, morph_operation.type, kernel, iterations=iteration)
                # Recreate potential deleted bacteria
                self.reconstruct(image, markers)
            else:
                image = cv2.morphologyEx(image, morph_operation.type, kernel, iterations=iteration)
        
        return image

    def apply_kmean(self, image: Mat, output_bundle: ResultBundle) -> Tuple[Mat, Mat]:
        """Apply the k-mean algorithm to:
        * Separate an image in 3 channels
        * Extract the median channel as a mask
        * Eventually aplly a morph operation on the channel
        * Replace this channel by backround color in the original image

        Args
        ----
            image: Mat - The image in which to apply the kmeans
            hyper_parameter_bundle: HyperParametersBundle - The hyper parameters bundle that contains tuning parameter values
            output_bundle: ResultBundle - Bundle in which to report results

        Returns
        ----
            Output bundle state will be modified.
            Tuple[Mat, Mat]: suppressed image, binarized image
        """
        #Kmeans
        working_image = image.copy()
        image_as_array = np.float32(working_image.reshape((-1,3)))
        _, label, center = cv2.kmeans(image_as_array, self.K, None, self.CRITERIA, self.ATTEMPTS, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)

        #Find index of min/aver/max channels
        averages = [np.mean(c) for c in center]
        min_index = np.argmin(averages)
        max_index = np.argmax(averages)
        med_index = list(set((0,1,2)) - set((min_index, max_index)))[0]

        #Suppress medium center directly in center array
        center[med_index] = center[max_index] #Replace by background

        res = center[label.flatten()]
        suppressed_image = res.reshape((image.shape))
        output_bundle.suppressed_image = suppressed_image
        
        #Create 1D binary mask that is active only on medium channel
        x,y,_ = image.shape
        label = label.reshape((x,y))
        binary_mask = np.uint8(np.vectorize(lambda x: 1 if x==med_index else 0)(label))
        output_bundle.kmean_mask_image = binary_mask

        #Binarize supressed image to obtain binarized image
        gray_image = cv2.cvtColor(suppressed_image, cv2.COLOR_RGB2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (5,5), 0)
        _, binarized_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        output_bundle.kmean_bin_image = binarized_image

        return suppressed_image, binarized_image
    
    def apply_distance_tranform(self, binarized_image: Mat, hyper_parameter_bundle: HyperParametersBundle, output_bundle: ResultBundle) -> Mat:
        """Apply the distance transform on the given binary image

        Args
        ----
            binarized_image: Mat - The binarized image in which to apply the distance transform
            hyper_parameter_bundle: HyperParametersBundle - The hyper parameters bundle that contains tuning parameter values
            output_bundle: ResultBundle - Bundle in which to report results

        Returns
        ----
            Output bundle state will be modified.
            Mat: A binarization of the distance transform image
        """        
        #Apply dist transform and normalize
        dist_transform = cv2.distanceTransform(binarized_image, cv2.DIST_L2, 3, dstType=cv2.CV_8U)
        dist_transform = cv2.normalize(dist_transform, np.zeros_like(dist_transform), 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #Dist transform morph operation
        dist_morph = hyper_parameter_bundle.dist_transform_morh
        dist_transform = self.apply_morphology(dist_transform, hyper_parameter_bundle.morph_kernel, dist_morph, hyper_parameter_bundle.repair_lost_baterias)
        output_bundle.distance_image = dist_transform
    
        # Threshold dist transform 
        thresh_min, thresh_max = hyper_parameter_bundle.dist_threshold_range
        _, distance_image = cv2.threshold(dist_transform, thresh_min*255, thresh_max*255, cv2.THRESH_BINARY)
        output_bundle.distance_bin = distance_image
        # apply yet anoter morph
        binarized_morph = hyper_parameter_bundle.binarized_dist_morph
        distance_image = self.apply_morphology(distance_image, hyper_parameter_bundle.morph_kernel, binarized_morph, hyper_parameter_bundle.repair_lost_baterias)
        output_bundle.distance_bin_morph = distance_image

        return distance_image
    
    def create_markers(self, binary_image: Mat, output_bundle: ResultBundle) -> Tuple[Mat, int]:
        """Create the markers from a distance image. The number of bacterias is known at this step.

        Args
        ----
            distance_image: Mat - A binarized image from which to create the markers
            hyper_parameter_bundle: HyperParametersBundle - The hyper parameters bundle that contains tuning parameter values
            output_bundle: ResultBundle - Bundle in which to report results

        Returns
        ----
            Output bundle state will be modified.
            Mat: The markers
            int: The number of bacterias
        """        
        dist_8u = binary_image.astype('uint8')
        contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bacteria_count = len(contours)

        # Create the marker image with a different color for each
        markers = np.zeros(binary_image.shape, dtype=np.int32)
        for i in range(len(contours)):
            cv2.drawContours(markers, contours, i, (i+1), -1)
        cv2.circle(markers, (5,5), 3, (255,255,255), -1)

        #Enhance colors and put as 8bits for display
        markers_8u = (markers * 10).astype('uint8') 
        output_bundle.marker_image = markers_8u
       
        return markers, bacteria_count

    def apply_watershed(self, original_image: Mat, suppressed_image: Mat, markers: Mat, marker_count: int, output_bundle: ResultBundle) -> None:
        """Apply the watershed algorithm on the given image & markers in order to have a nice result image. Optional step, useless for count.

        Args
        ----
            original_image: Mat - Used to annotate and put on the result bundle
            suppressed_image: Mat - The image in which to apply the watershed on
            markers: Mat - The markers for the watershed
            marker_count: int - Number of markers
            hyper_parameter_bundle: HyperParametersBundle - The hyper parameters bundle that contains tuning parameter values
            output_bundle: ResultBundle - Bundle in which to report results

        Returns
        ----
            Output bundle state will be modified.
        """
        # Perform the watershed algorithm
        watershed_result = cv2.watershed(suppressed_image, markers)

        # Generate random colors
        colors = []
        for i in range(marker_count):
            colors.append((randint(0,255), randint(0,255), randint(0,255)))

        # Create the result image
        watershed_image = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
        # Fill labeled objects with random colors
        for i in range(markers.shape[0]):
            for j in range(markers.shape[1]):
                index = markers[i,j]
                if index > 0 and index <= marker_count:
                    watershed_image[i,j,:] = colors[index-1]
        
        output_bundle.watershed_image = watershed_image

        # Show contour on original image image
        result_image = original_image.copy()
        result_image[watershed_result == -1] = [255,0,0]
        output_bundle.annotated_image = result_image
    
    def compute_diff(self, count: int, expected_count: List[int]) -> int:
        """Compute the difference between a count and a range of expected count.
        
        Returns
        ---
            The difference: 
                * 0 if count is in expected counts
                * Difference between the count and the min/max of the list otherwise
        """
        if count in expected_count:
            diff = 0
        elif count > max(expected_count):
            diff = count - max(expected_count)
        else:
            diff = min(expected_count) - count
        return diff

    def process(self, image_name: str, image: Mat|None, hyper_parameter_bundle: HyperParametersBundle) -> ResultBundle:
        start_time = time.time()
        output_bundle = ResultBundle()
        output_bundle.image_name = image_name
        image_label = self.labels["images"][image_name]

        #Load image if necessary
        if image is None:
            image_filepath = os.path.join(os.path.dirname(__file__) , f"{self.IMAGES_FILEPATH}/{image_name}")
            image = cv2.cvtColor(cv2.imread(image_filepath), cv2.COLOR_BGR2RGB)
        output_bundle.original_image = image
        
        ### K-means ###
        suppressed_image, binarized_image = self.apply_kmean(image, output_bundle)

        ### Distance transform ###
        distance_image = self.apply_distance_tranform(binarized_image, hyper_parameter_bundle, output_bundle)

        # Create the marker images
        markers, bacteria_count = self.create_markers(distance_image, output_bundle)

        label_bacteria_count: list[int] = image_label["bacteryCount"]
        output_bundle.count = bacteria_count
        output_bundle.expected_count = label_bacteria_count
        output_bundle.diff = self.compute_diff(bacteria_count, label_bacteria_count)

        ### Optional- WATERSHED ###
        self.apply_watershed(image, suppressed_image, markers, bacteria_count, output_bundle)

        output_bundle.runtime_s = round((time.time() - start_time), 3)

        return output_bundle

if __name__ == "__main__":
    algo = AlgorithmProcessor()

    from data_classes import HyperParametersBundle

    hyp = HyperParametersBundle(
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), #Morph kernel options
            True, #Whether to repair bacterias

            #Dist morphs
            MorphOperation(0, None),
            
            #Bin morphs
            MorphOperation([1,2,3,4,5], cv2.MORPH_ERODE), 

            (0.0, 1.0), #Dist threshold values
        )

    res = algo.process("Candida.albicans_0004.png", None, hyp)

    res.figure_all()
    import matplotlib.pyplot as plt
    plt.show()

    print(res.runtime_s)