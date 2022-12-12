import cv2
from data_classes import MorphOperation
from cross_search import HyperParametersBundleMatrix
import cross_search

def test_single_thread(matrix: HyperParametersBundleMatrix):
    cross_search.search_matrix(matrix)

def test_multiple_thread(matrix: HyperParametersBundleMatrix, nb_procs: int):
    cross_search.search_matrix_parallel(matrix, nb_procs)

if __name__ == "__main__":
    print(f"Starting algorithm")

    dist_morphs = [[2], [2, 2], [1, 1, 1, 1, 1, 1]]
    bin_morph = [[1, 1, 1, 2, 2, 2, 2, 1, 1, 1], [1, 1, 2, 2, 2, 2, 2, 2, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]

    hyper_parameters_search_matrix = HyperParametersBundleMatrix(
        [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))], #Morph kernel options
        [True], #Whether to repair bacterias

        #Dist morphs
        [MorphOperation(0, None)]
            + [MorphOperation(i, cv2.MORPH_OPEN) for i in dist_morphs],

        #Bin morphs
        [MorphOperation(i, cv2.MORPH_ERODE) for i in bin_morph],

        [(min_r, 1.0) for min_r in [0.0, 0.2]], #Dist threshold values
    )

    test_multiple_thread(hyper_parameters_search_matrix, 4)