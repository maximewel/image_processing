Name: Geminiani Maxime
Github: https://github.com/maximewel/image_processing

Brief desc: Assignement 1's code is at the root of this folder (1_flippity_floppity.ipynb). The image ised as input is in "in", and the ouput is in "out. 
Technology: I use OpenCV as my image processing framework, with numpy as my mathematical engine and matplotlib as my display.
I run that into a jupyter notebook as the format is adapted for quick exercise-related one-shot development.

Algo: The algorithms are explained in the notebook quickly. they are perfomed on a 3D Matrix (Or a 2D matrix of pixels, each pixel composed of a single or multiple channels).
The horizontal and vertical flips are performed with python slicing methods. The flipping of both is done via a double loop to show that each pixel can be manipulated individually.
Inversing a pixel is simply switching its coordinate with its symetrical with respect to the reference axis: with a loop, this is X-i with i the coordinate of the pixel and 
X the maximum index of the pixel. In our case, the max index is the size returned by the shape object minus 1 (512 pixels are indexed 0-511).

Note: For further questions, you can contact me. If the jupyter notebook + comments are sufficient and easy to read for you, I can include this text file in the notebook in further assignments.