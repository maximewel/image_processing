import cv2
from algo import AlgorithmProcessor, HyperParametersBundle, ResultBundle, MorphOperation
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    print(f"Starting algorithm")

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    o1 = MorphOperation(0, cv2.MORPH_DILATE)
    o2 = MorphOperation(0, cv2.MORPH_OPEN)
    o3 = MorphOperation(1, cv2.MORPH_OPEN)
    hyper_parameters = HyperParametersBundle(k, o1, o2, o3)
    algorithm_processor = AlgorithmProcessor()

    for image_name in ["Candida.albicans_0004.png", "Candida.albicans_0008.png", "Candida.albicans_0018 copy.png"]:
        
        res: ResultBundle = algorithm_processor.process(image_name, hyper_parameters)
        print(f"{res.image_name}: diff={res.diff} Expected {res.expected_count}, got {res.count}")

        images = [    
            res.original_image,
            res.kmean_image,
            res.suppressed_image,
            res.kmean_bin_image,
            res.distance_image,
            res.distance_bin,
            res.distance_bin_2,
            res.marker_image,
            res.watershed_image,
            res.annotated_image,
        ]

        image_names = [    
            "res.original_image",
            "res.kmean_image",
            "res.suppressed_image",
            "res.kmean_bin_image",
            "res.distance_image",
            "res.distance_bin",
            "res.distance_bin_2",
            "res.marker_image",
            "res.watershed_image",
            "res.annotated_image",
        ]
        
        fig, ax = plt.subplots(1, len(images))
        for i, im in enumerate(images):
            ax[i].set_axis_off()
            ax[i].set_title(image_names[i])
            if len(im.shape) == 3:
                ax[i].imshow(im)
            else:
                ax[i].imshow(im, cmap="gray", vmin=np.min(im), vmax=np.max(im))

        plt.show()