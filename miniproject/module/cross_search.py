from algo import AlgorithmProcessor, HyperParametersBundle, MorphOperation, ResultBundle

from typing import List, Tuple
from dataclasses import dataclass
import sys
import os
import numpy as np
import cv2
from rich.progress import Progress, MofNCompleteColumn, TimeElapsedColumn, TextColumn, BarColumn

IMAGES_FILEPATH = "../dataset/png"

@dataclass
class HyperParametersBundleMatrix:
    """A single morph kernel is used throughought the project in order to reduce the
    number of kernel possibilities"""
    morph_kernel_options: List[Tuple] #Recommended: See cv2.getStructuringElement

    kmean_median_channel_morph_options: List[MorphOperation]
    dist_transform_morh_options: List[MorphOperation]
    binarized_dist_morph_options: List[MorphOperation]

    #Threshold to apply on the normalized distance image between [0.0, 1.0]
    dist_threshold_range_options: List[Tuple[float,float]]

    def size(self) -> int:
        """Return the product of the sum of all the options of this matrix"""
        return np.prod([len(option_list) for option_list in [
            self.morph_kernel_options,
            self.kmean_median_channel_morph_options,
            self.dist_transform_morh_options,
            self.binarized_dist_morph_options,
            self.dist_threshold_range_options,
        ]])

def load_image(image_name: str) -> np.ndarray:
    image_filepath = os.path.join(os.path.dirname(__file__) , f"{IMAGES_FILEPATH}/{image_name}")
    image = cv2.cvtColor(cv2.imread(image_filepath), cv2.COLOR_BGR2RGB)
    return image

def bundle_generator(matrix: HyperParametersBundleMatrix):
    """Generator object used to avoid having every bundle in memory"""
    for morph_kernel in matrix.morph_kernel_options:
        for kmean_morph in matrix.kmean_median_channel_morph_options:
            for dist_morph in matrix.dist_transform_morh_options:
                for bin_morph in matrix.binarized_dist_morph_options:
                    for dist_treshold in matrix.dist_threshold_range_options:
                        yield HyperParametersBundle(morph_kernel, kmean_morph, dist_morph, bin_morph, dist_treshold)

def search_matrix(matrix: HyperParametersBundleMatrix):
    """Search every choice of the hyper parameter bundle matrix. Compare the differences and return the best one"""

    #Create generator that is able to create bundles and yield them when necessary, not wastime any memory
    hyper_parameter_bundles = bundle_generator(matrix)
    
    #As we have a relatively small dataset, take eveything in memory to reduce IO operations
    image_filepath = os.path.join(os.path.dirname(__file__), f"{IMAGES_FILEPATH}")
    image_names = os.listdir(image_filepath)
    image_name_bin = [(image_name, load_image(image_name)) for image_name in image_names]

    image_count = len(image_names)
    bundle_count = matrix.size()

    algorithm_processor = AlgorithmProcessor()
                
    print(f"Testing {bundle_count} hyper-parameter bundles on {image_count} images")

    min_diff = sys.maxsize
    best_hyper_parameters: HyperParametersBundle = None
    with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()) as progress:
        bundle_progress = progress.add_task("[blue]Processing bundle...", total=bundle_count)
        image_progress = progress.add_task("[green]Processing image...", total=image_count)
        for bundle in hyper_parameter_bundles:
            progress.reset(image_progress)
            #Comopute on each image
            result_bundles = []
            for image_name, image in image_name_bin:
                res = algorithm_processor.process(image_name, image, bundle)
                result_bundles.append(res)
                progress.advance(image_progress)
            #Do total diff
            total_diff = sum([bundle.diff for bundle in result_bundles])
            #Compare to best
            if total_diff < min_diff:
                min_diff = total_diff
                best_hyper_parameters = bundle
            progress.advance(bundle_progress)

    print(f"Finished, best hyp bundle: diff={min_diff} with hyper-parameters \n{best_hyper_parameters}")