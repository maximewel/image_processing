Name: Geminiani Maxime\
Github: https://github.com/maximewel/image_processing/tree/main/assignement_4\
Jupyter notebook to see the code and results directly in github: https://github.com/maximewel/image_processing/blob/main/assignement_4/convolutions.ipynb\
\
Brief desc: Assignement 4's objective is to play with convolutions in the scope of mean filters, edge detection filters, and statistical filters.

Where to find the files:
The work is done in the jupyter notebook
The /out folder has:

* Ex1: means
  * Image with mean filter 3\*3, 5\*5, 9\*9: mean_filters_3_5_9.png
  * Image that compares the border extensions done in this notebook: 1_border_extensions.png
  * Image that compare the means with the different border extensions: 2_mean_border_extension_diff.png
* Ex2: Grad-LoG
  * Image that do the gradient_laplacien of an image:
    * 3_0_gradient_laplacien_Ara-grey.png
    * 3_1_gradient_laplacien_Lena-grey.png
    * 3_2_gradient_laplacien_Stop-grey.png
* Ex3: Statistical Min, Max
  * 3\*3 neighbours: 4_min_max_Lena-grey_3_3.png
  * 5\*5 neighbours: 5_min_max_Lena-grey_5_5.png


# explain how you get the same input and output size
I do the padding myself and not use the cv2.pad() method so I can explain exactly what happens. Here is the algorithm I use:

* The filter_image receives an image W*H as the original image
* The filer_image receives a kernel of size Wk, Hk
  * It knows that it has indetermined border values for Wk-1 pixels: Assuming the kernel is odd (3,5,7), it will had 2,4,6 pixels (1/2/3 at each of the 4 sides of the picture)
* An extended copy of the original image is created that is determined at these values: It is of size (W+(Wk-1 /2), H+(Hk-1 /2))
  * For a simple kernel of size (3,3), a pixel is added at each border of the image
  * This is not done in place. This means a copy of the image is created and it is not "extended" in place. This is a design choice that sacrifies space complexity for usability as the original image is unchanged, but with a heavy dataload, this could be done in place or with an additional "layered" picture with only the border
* The padding is important: Pixels are added, but of what intensity ? The script proposes three easy methods form the course
  * Blank: All pixels are 0
  * Extended: All pixels have the value of the last border
  * Mirrored: Pixels are mirrored with the border as the symetry axis
* Finally, the resulting image is computed
  * The resulting image has the same size as the original image
  * The extended image is only used for computations by the apply kernel function
  * The additional borders are never present in the resulting images, they are only usefull for determining behavior at the border
  * An exemple of the 3 borders and their results is presented and briefly discussed in the notebook, an image is present on the output folder

# a short description of you gradient filter and the Laplacian of Gaussian filter
* The LoG filter is done via the convolution kernel present in the course
  * It could be separated for less time complexity
  * It could be bigger / smaller depending on the needs
* The gradient filter is done in two steps
  * Horizontal gradient, kernel from the course formula
  * Vertical gradient, kernel from the course formula
  * Addition via cv2.max that keeps intensity of both images (regular ndarray addition does a modula, and thus loses high intensity information)
  * For the gradient filter, I had to add a "abs" value at the end of my kernel of I would have had negative values in my images. I could also truncate them to 0.
  * kernels are 1D but are coded in 2D as my apply_kernel function always expects a 2D kernel

# and the responses to the questions 1. c), 2. d) et 3. c).
The responses are in context in the notebook and reported below

## 1.c) Comment on the results obtained with the different filter sizes

### Kernel size
The mean filters are more impactfull the bigger the kernels are: the resulting intensity is more and more dilluted.\
This results in a longer processing time and a more powerfull blur effect.\
This is a low-pass filter: The details (high-intensity signals) are lost and the resulting image becomes harder to see and tends to a single shade of grey.

### Padding type
The border extension of the images implemented in this notebook are blank / extended / mirrored. The figure 2 shows their effects.\
There is a clear effect of blank padding on images are it tends to add a black frame at the border as the blank pixels are black (intensity 0). Even on a mean filter of 11, the extended and mirror padding don't make a great difference on the pictures tested (Ara, Lena, Baboon).

## 2.d) Comment on the results obtained with the two methods

* The gradient edge detection works in two steps that can be combined as shown above. The result is an image that has sharp lines and very precise edges around great changes in intensity.
* The Laplacian of Gaussian works in one step but has a bigger kernel - and the kernel used above is not he biggest one that can be computed. The filter's resulting image seems to separate the background in black from the principal shapes that are activated in white. 

The gradiant methods is done in vertical / horizontal steps, meaning they have a greatly reduced complexity if done right. If the Laplacian kernel is separable (it sure seems like it), both methods would be similar in complexity.\
The gradiant filters extract sharp edges (lines) separating objects whereas the Laplacian filter activates shapes that are filled - in order to have the same result, we would have to either apply a region mecanism for the gradiants or only take the edges for the Laplacian All in all, both produce great results and are already extracting most of the simple shapes of the three images tested (Lena, Stop, Ara)

## 3.c) Comment on the results obtained with the individual filters and their combination

### Max, Min, MaxMin

* Max filter: The max filter makes the image very white and thus exacerbate the intense parts of the image by making them multiple pixels wide, depending on the kernel size
* Min filter: Similarly, the image becomes black and the min filter exacerbates the low parts of the image by making them multiple pixel wide

In both cases, the homogenous parts of the image stay the same as a min/max has basically no effect. But every low-intensity region is amplified by the min and every high-intensity region is amplified by the max filter.

* Max - Min filter: The resulting image of Max - min is very similar to the Horizontal + Vertical addition of the gradient filter: It has sharp edges as high-intensity pixels.
    * Effectively, every homogenous part of the image is both in the max and min - and thus removed from the last image. Every part where there is a high max & low min or high min & low max is amplified - and every part that is changing from max to min and vice-versa is an edge.

### Neighbourood size
The neighbourood size testes are 3\*3 and 5\*5. Although the 3\*3 filter shows great results, the 5\*5 filter is lacking in the sense that has a blurring effect. This could be because the min/max neighbourood being too great displaces the min/max values too far out of their regions. The resulting Max-Min image is also blurred and the edges are less sharp and ill defined.