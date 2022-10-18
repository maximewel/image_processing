Name: Geminiani Maxime
Github: https://github.com/maximewel/image_processing/tree/main/assignement_3

Brief desc: Assignement 3's objective is to be able to equalize an histogram on a grayscale image, separate the r,g,b components of an RGB image, and transfer it to the HSL domain. Using these tools, the last part of the assignement is to do an histogram equalization on each R,G,B channel and rebuild the image, and compare it to the same image encoded in HSL and with the L channel equalized.

Where to find the files:
The work is done in the jupyter notebook
The /out folder has:

* Three examples of grayscale image equalization with equalized_(Baboon, lena, Beetle).jpg
* The original image equalized in RGB and HSL (L channel) domains as hsl_rgb_equalisations_results.jpg
* The original image with its RGB and re-converted HSL equalizations's histograms over the 3 channels as hsl_rgb_equalisations_histograms

Algo: The algorithm is explained in depth in the jupyter notebook and is rather simple.
* The HSL_to_RGB and reverse algorithm are directly taken from the source given in the assignement document.
* The histogram equalization applies the course' formula:
  * From a one-channel (2D) image, an histogram is made
  * From the histogram, the cumulative histogram can be computed
  * Using the cumulative histogram, a convertion function can be created that sends each pixel of the original image to its value on the cumulative hist divided by the size of the image. This real value can then be quantified, generating the value of the pixel on the equalized image.

Another interesting tricked using during the HSL equalization :
The L channel is effectively a real value between 0 and 1. Therefore, the regular histogram function can not be performed easily as it expect integer values from 0 to 256. Instead of adapting this function to accept real numbers (ex: storing the count in a dictionnary that accepts real as keys instead of an array that can only be indexed by integer values), the quant value has been used.
* First, the L channel is passed through the quant function that transforms the valus from a real domain [0,1] to an integer domain [0,255] 
* Secondly, the equalization is done using this discrete integer channel
* Finaly, the resulting equalized channel is passed through the real function again to go back to a [0,1] domain


Note: For further questions, don't hesitate to contact me. You can play with the jupyter notebook to observe the algorithm on different images of the input folder.