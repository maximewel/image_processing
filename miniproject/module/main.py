import cv2
from algo import MorphOperation
from cross_search import HyperParametersBundleMatrix
import cross_search

if __name__ == "__main__":
    print(f"Starting algorithm")

    hyper_parameters_search_matrix = HyperParametersBundleMatrix(
        [cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))], #Morph kernel options

        [MorphOperation(0, None)] + [MorphOperation(iter, cv2.MORPH_DILATE) for iter in range(1, 3)], #Kmean morphs
        [MorphOperation(0, None)] + [MorphOperation(iter, cv2.MORPH_OPEN) for iter in range(1, 3)] \
            + [MorphOperation(iter, cv2.MORPH_ERODE) for iter in range(1, 3)], #Dist morphs
        [MorphOperation(0, None)] + [MorphOperation(iter, cv2.MORPH_OPEN) for iter in range(1, 3)] \
            + [MorphOperation(iter, cv2.MORPH_ERODE) for iter in range(1, 3)], #Bin morphs

        [(min_r/100, 1.0) for min_r in range(30, 60, 5)], #Dist threshold values
    )

    hyper_parameters_search_matrix = HyperParametersBundleMatrix(
        [cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))], #Morph kernel options

        [MorphOperation(0, None), MorphOperation(1, cv2.MORPH_DILATE)], #Kmean morphs
        [MorphOperation(0, None)] + [MorphOperation(iter, cv2.MORPH_OPEN) for iter in range(1, 3)] \
            + [MorphOperation(iter, cv2.MORPH_ERODE) for iter in range(1, 3)], #Dist morphs
        [MorphOperation(iter, cv2.MORPH_OPEN) for iter in range(1, 3)] \
            + [MorphOperation(iter, cv2.MORPH_ERODE) for iter in range(1, 3)], #Bin morphs

        [(min_r/100, 1.0) for min_r in range(40, 55, 5)], #Dist threshold values
    )

    cross_search.search_matrix(hyper_parameters_search_matrix)