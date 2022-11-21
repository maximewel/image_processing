# Pre-project initialisation
This section initializes the project by stating the administrative properties of the project, and goes on with a preliminary design discussion about the challenges / approach taken.

## Administrative

* Project: Counting bacteria
* Name: Maxime Welcklen
* Github link: https://github.com/maximewel/image_processing/tree/main/miniproject
* Technology: CV2 (Python)

## Preliminary design

> - your understanding of the problem (what are the challenges, what will be easy...) with preliminary ideas how to solve it.

## Goal/dataset

To understand the project, the most important things to analyze are the goal and the dataset used. For this purpose, the dataset can already be found in the ./dataset folder. \
The goal of the project is to produce a script that takes a bacteria image as input and, thanks to image processing techniques saw in the class and other related functionalities, is able to output the number of bacteria in the picture. \
The initial dataset is an extract of a larger dataset and is composed of 10 images of bacterias in the scope of tens and seems always less than a hundred.\


(+)  From the dataset, we can anticipate some advantages:
* The bacteria appear to be fairly well separated, with little overlap.
* Separated color domains: bacteria and backgrounds colors are consistent between images.
* Non-disruptive background: The background is a clear color that does not appear to cause issues (a reflective surface can be a pain with luminosity).


(-) Of course, there are also a few challenges in view:
* Irregular shape: The bacteria are between round and ellipsis-shaped, with many irregular shapes.
* Close together: Although they don't overlap, the bacteria often touch each other.
* Blurriness: Images have some blurry effects that can greatly affect results without sharpening.

The goal of this project is not only to be "perfect on all 10 images." Firstly, because the dataset is a reduced one, having a very precise algorithm on 10 images is worthless if it is incapable of working on any new image that would be similar. Of course, a new image should conform to the standards assumed (a sudden image of triangle bacteria would be incapable to process), but overfitting the dataset is not the goal. Furthermore, I would rather develop a discussion aside from the algorithm itself with tries and fails and an analysis of multiple approaches with multiple techniques.

## Start of Reflections
Among the techniques that I find interesting for this project, there are:

*Directly modified/ported from the course*
* **Histogram analysis**: Simply starting by discussing the histogram for a bit might help understand the dataset better (are the colors really that well separated ? How many different modes can we find ?).
* **Binarisation**: Binarization is very good at separating in black/white two zones in the images. As the background is in a very specifc color tone, binarisation could help differentiate the data from it. I am not sure it would be usefull, but it could be a good part of the pre-processing (before detecting the cells).
    * See segmentation as a 'multi-zone' binarisation that could be more adapted.
* **Edge detection [Kernel]**: Likewise, kernel-based edge detection - that could be done via the techniques we saw in the course, such as min/max, LoG... - could help have very sharp and clear bacteria shapes and remove the somewhat blurry effect of the dataset.

*From research/framework/past work*
* **Edge detection [Other]**: other edge detections methods - based on Hough or Canny - can be used similarly as kernel-based ones and their efficiency can be compared.
* **Open/close**: These are synthetic procedures built from erosion/dilatations on an image. They are used to respectively close gaps or remove protusions in the shapes of the image. A combination of close then open can do wonders toobtain more separated and "smoothed" shapes.

I also see techinques in order to do the last counting:
* **Segmentation**: Segmentation is separating the images in differentiated zones. Similar to binarization, but with multiple regions in the image. This is often done after a pre-processing step and an edge detection process, and could be done as one of the "last steps.". With segmentation, we could, for example, try to count the different zones found and subtract the background.
* **Pattern matching**: CV2 has a good pattern matching module that can help find shapes such as ellipses or circles. It could be used as a "last step" in the script.

Finally, I see techniques that I don't see as useful at all:
* *Equalization*: The color tones seem well defined/separated, so I intuitively see equalization as a tool that could be counterproductive for this project's purpose: From separated color tones, it would "spread" the pixel values over the whole RGB domain/histogram, and thus it could lose the separation between the classes (inner cell, outer cell, background, etc.).
* *Blurring*: Blurring an image with a mean or a gaussian blur is a step that, from my experience, can be used as a pre-processing step when manipulating unclean images with different luminosities in order to attenuate the sudden color changes due to shades/illumination (or imperfections from the sensor). I feel like the bacteria dataset has the opposite needs - it is rather blurry, and the script would likely want to sharpen the shapes in order to have clear separated bacteria, not blurry ones.

The goal will be to find the best parameters adapted to the solution without being too influenced by the dataset itself, as well as to try different combinations of image analysis techniques that produce the best results.