Name: Geminiani Maxime\
Github: https://github.com/maximewel/image_processing/tree/main/assignement_6\
Brief desc: Assignement 6's objective is to compress an image choosing an algorithm compression, with run-length / quadtree being proposesd as default

Where to find the files:
The work is done in the jupyter notebook
The /out folder has:

* serialized_page-SD3-1003.bin: The SD3-1003 image saved in binary
* 0_iterative_quad_tree_page-SD3-1003.png: Image showing the building of the quadtree iteratively
* 1_iterative_differences_page-SD3-1003.png: Image showing the difference between each iteration of the quadtree with a growing depth (i+1 - i)
* 2_iterative_activation_map_page-SD3-1003.png: Image showing the nodes activated at each depth iteration:
  * In white, the nodes already activated in precedent iterations
  * In red, the nodes activated at this depth (means they are encoding their patches and don't need to be defined further, translating into a compression of the encoding)
  * In black, the part of the image that is not activated yet (so no encoding yet)
* 3_rebuilt_imagepage-SD3-1003.png: Image transformed to a quadtree, serialized to a binary string and re-built as a quad-tree.


# brief explanation of your algorithm
This compression is done via a QuadTree algorithm: The image is separated in partitions. Each partition either contains a uniform color, or is split between 4 sub-partitions untill a uniform patch is found (with a min size of 1*1 being a pixel).\
The implementation is based around two classes: 
* A QuadTree class offers the basic meta-function for manipulating the quadtree with its image (ask to build the tree from image, serialize it, and de-serialize it) as well as offer some nice to have display function (Ex: Returning the tree at a certain depth). A Quadtree contains QuadTreeNodes.
* The QuadTreeNode is a recursive class containing either a color or childs and a mean color. The quadtree node is capable of creating itself and its children from an image patch and to serialize/deserialize itself from a byte array. It also computes its mean color at initialization if it has children: This allows an easy way to show intermediate Quadtree.

# 3. Compression result
##  (a) For the chosen compression strategy, calculate the number of bits needed to represent the sample image after compression.

* Original RGB Image size: 430.08 kb
* Quadtree stored binary size: 82.702kb
* Ratio: The size is reduced by a factor or 5.2
* New storage takes only 19.0% of RGB storage