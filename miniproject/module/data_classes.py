
from __future__ import annotations
#Image proc imports
import matplotlib.pyplot as plt
import numpy as np
from cv2 import Mat
from typing import Tuple, List

#Other imports
from dataclasses import dataclass

@dataclass
class HyperParametersBundle():
    """A single morph kernel is used throughought the project in order to reduce the
    number of kernel possibilities"""
    morph_kernel: tuple #Recommended: See cv2.getStructuringElement
    repair_lost_baterias: bool

    dist_transform_morh: MorphOperation
    binarized_dist_morph: MorphOperation

    #Threshold to apply on the normalized distance image between [0.0, 1.0]
    dist_threshold_range: Tuple[float,float]

class ResultBundle():
    image_name: str 

    #Images
    original_image: Mat
    kmean_mask_image: Mat
    suppressed_image: Mat
    distance_image: Mat
    distance_bin: Mat
    distance_bin_morph: Mat
    marker_image: Mat
    watershed_image: Mat
    annotated_image: Mat
    kmean_bin_image: Mat

    #Quantified results
    count: int
    expected_count: list[int]
    diff: int
    runtime_s: float

    def summary(self):
        """Create a plot figure summarizing the result bundle"""
        fig, axes = plt.subplots(1, 4)

        fig.text(0.1, 0.1, f"Differences: {self.diff}, counted {self.count}/{self.expected_count} in {self.runtime_s}s")

        axes[0].set_axis_off()
        axes[0].set_title("Original image")
        self.plt_adaptative_imshow(axes[0], self.original_image)

        axes[1].set_axis_off()
        axes[1].set_title("Marks image")
        self.plt_adaptative_imshow(axes[1], self.marker_image)

        axes[2].set_axis_off()
        axes[2].set_title("Watershed image")
        self.plt_adaptative_imshow(axes[2], self.watershed_image)

        axes[3].set_axis_off()
        axes[3].set_title("Final image")
        self.plt_adaptative_imshow(axes[3], self.annotated_image)

        fig.tight_layout()

        return fig

    def figure_all(self):
        """Return a plot figure containing all images of the result bundle, one by one"""
        kmean_images = [
            ("K-mean mask", self.kmean_mask_image),
            ("Supressed image", self.suppressed_image),
            ("Binarized K-means image", self.kmean_bin_image)
        ]

        distance_images = [
            ("Distance image", self.distance_image),
            ( "Binarized distance image", self.distance_bin),
            ("Morphed binary distance", self.distance_bin_morph)
        ]

        watershed_images = [
            ("Marker image", self.marker_image),
            ("Watershed results", self.watershed_image),
            ("Annotated image", self.annotated_image),
        ]
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 15))
        fig.suptitle(f"Result images for image {self.image_name}", fontweight="bold", fontsize="x-large")
        subfigs = fig.subfigures(nrows=4, ncols=1)

        #Place orig image
        subfig = subfigs[0]
        subfig.subplots_adjust(top=0.75)        
        axes = subfig.subplots(nrows=1, ncols=3)
        axes[0].set_axis_off()
        axes[1].set_title("Original image")
        axes[1].set_axis_off()
        axes[1].imshow(self.original_image)
        axes[2].set_axis_off()

        #Place Kmeans
        subfig = subfigs[1]
        subfig.subplots_adjust(top=0.75)
        subfig.patch.set_facecolor('grey')
        subfig.suptitle("Kmean processing", fontweight="bold", fontsize="large")
        kmean_ax = subfig.subplots(nrows=1, ncols=3)
        for i, (name, im) in enumerate(kmean_images):
            kmean_ax[i].set_axis_off()
            kmean_ax[i].set_title(name)
            self.plt_adaptative_imshow(kmean_ax[i], im)

        #Place dist
        subfig = subfigs[2]
        subfig.subplots_adjust(top=0.75)
        subfig.suptitle("Distance transform processing", fontweight="bold", fontsize="large")
        dist_ax = subfig.subplots(nrows=1, ncols=3)        
        for i, (name, im) in enumerate(distance_images):
            dist_ax[i].set_axis_off()
            dist_ax[i].set_title(name)
            self.plt_adaptative_imshow(dist_ax[i], im)

        #Place watershed
        subfig = subfigs[3]
        subfig.subplots_adjust(top=0.75)
        subfig.patch.set_facecolor('grey')
        subfig.suptitle("Watershed processing", fontweight="bold", fontsize="large")
        watershed_ax = subfig.subplots(nrows=1, ncols=3)
        for i, (name, im) in enumerate(watershed_images):
            watershed_ax[i].set_axis_off()
            watershed_ax[i].set_title(name)
            self.plt_adaptative_imshow(watershed_ax[i], im)

        return fig
    
    def plt_adaptative_imshow(self, ax, im):
        if len(im.shape) == 3:
            ax.imshow(im)
        else:
            ax.imshow(im, cmap="gray", vmin=np.min(im), vmax=np.max(im))

@dataclass
class MorphOperation():
    iterations: int | List[int]
    type: int #cv2.MORPH_XXX (Open, close, dilate, erode)