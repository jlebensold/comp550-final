Fall 2018 COMP 550: 
Group Number: 19

-------------------------------------------------
Team Members
-------------------------------------------------
Jonathan Maloney-Lebensold (260825605)


-------------------------------------------------
Introduction
-------------------------------------------------

-------------------------------------------------
Datasets
-------------------------------------------------

* test_images_conn:
  The contour vectors computed from OpenCV (pickled) from the (unlabelled) test set.

-------------------------------------------------
Dependencies
-------------------------------------------------
Python 3.7

- numpy
- opencv
- pandas
- parfit
- Pillow
- pytorch
- scipy
- tensorboardX
- sklearn
- torchvision

-------------------------------------------------
Notes
-------------------------------------------------

The entry point is `main.py`. Different command-line arguments can be used to
run the pre-processing steps, train models and write kaggle prediction files.
The configuration parameters are found in `constants.py`.

See notebooks/ for our experiments

-------------------------------------------------
Example Usage
-------------------------------------------------

* Pre-process the images:
python main.py --pre-process

