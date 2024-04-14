# f20pa-anpr

Python and OpenCV solution to implement a standard ANPR solution, consisting of pipeline stages: image pre-processing, 
number plate extraction, character segmentation and character recognition.
predicted match. The main investigatory aim of this project is to assess the effectiveness of image
pre-processing Techniques in improving ANPR reading accuracy.

Note, this project only provides support for a specific computer-generated dataset (linked below).
Methods are optimised to process the images from `UKLicensePlateData`.
Significant efforts would be required in refactoring parameters of OpenCV methods for this system
to work on other sets of input images. 

Input dataset can downloaded from [here](https://www.kaggle.com/datasets/saadbinmunir/uk-licence-plate-synthetic-images/)

Libraries Required:
- `argparse`
- `sys`
- `cv2`
-  `os`
-  `time`
- `matplotlib`
- `numpy` 

---
4th Year Undergraduate BSc (Hons) Dissertation Project. John McIver. Heriot-Watt University, Edinburgh. School of
Mathematical and Computer Sciences. Department of Computer Science. jm2006@hw.ac.uk

All rights reserved. © 2024 John McIver.
Use of this project is granted, provided that proper reference and credit is provided to the author.


## Usage

This application is used via a command-line interface. The available arguments and purpose are listed below.

```
/usr/bin/python3 /Users/jmciver/PycharmProjects/f20pa-anpr/scripts/anpr.py --help
usage: anpr.py [-h] -d D [-l L] -s S [-p P]

optional arguments:
  -h, --help  show this help message and exit
  
  -d D        The directory containing the input image dataset.
  
  -l L        The number of files which should be processed from the input
              image dataset. Unspecified limit will default to processing all
              available images from the input directory.
              
  -s S        Specify the pre-processing pipeline stages which should be run
              against the dataset. 
              1a :- Noise Removal (Bilateral Filtering)
              1b :- Improving Contrast (Adaptive Histogram Equalisation) 
              1c :- Noise Removal (Adaptive Gaussian Thresholding) (on) | Default: Otsu's Thresholding
              1d :- Tilt Correction
              
  -p P        Plot and display the results of each pipeline processing stage.
              By default this will write result data to a directory
```

## Example Usage

`1a_1b_1c_1d_whiteplate_12000.xml` generated using the following command line arguments:

```
-d "/Users/jmciver/Documents/Y4S1/F20PA/DISSERTATION-MATERIAL/UKLicencePlateDataset/whiteplate_augmented" -l 12000 -s "1a 1b 1c 1d" -p false
```

Output XML file naming format specifies:
* `1a_1b_1c_1d_` correspond to the image pre-processing techniques that are enabled
* `_whiteplate_` corresponds to what input dataset the output processed xml file data is generated on
* `12000` specifies the number of images processed

Note, the internal XML parsing methods cannot be accessed through public functions - they are only run when a series of 
images are processed, returning the output summary.

## Code Architecture
### `anpr.py`
The main class which implements all critical ANPR functions, ingests input images and returns and displays a predicted 
result.


### `parse_data.py`
Helper class that outputs summary result metrics and data on input XML files. Note, only `parse_xml()` is publicly 
accessible to the user, this function is called once a directory of input images has been processed to generate 
a summary of resulting output data:

```python
def parse_xml(file_path):
    ...
    output_str = """+----------------+\n| Results Output |\n+----------------+\n Input File: {} \n
        Accuracy: {:0.5f}%
        Total processing time (sec): {:0.5f}
        Avg processing time per input (sec) : {:0.5f}
        Error rate: {:0.5f}% | {}/{}
        Incorrect Registrations (Predicted, Expected): {}
        Most Commonly Incorrect Characters (Actual character was misread as ... N times) | {}
        Average PSNR (dB) {:0.5f}
        Mean Confidence per Character {}
        """
    ...
```

The remaining methods of the class are backend private helper methods to generate specific metrics for evaluation, 
and are not public / user-facing. These methods are not required to be run as part of the base functionality of the 
system.

### `write_data.py`
Writer class that writes the data captured (in memory) from `anpr.py` to a formatted XML file.

## References
In-code citations are also listed above in function headers.

### `anpr.py`
`argparse` (accept CLI arguments) usage: https://docs.python.org/3/library/argparse.html

Convert an input RGB image to greyscale
- https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
- https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html (BGR2GRAY)


Bilateral filtering:  https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

Adaptive Histogram Equalisation
- https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/
- https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html

Adaptive Gaussian Thresholding + Otsu's Method
-  https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

Tilt Correction
- cv2.minAreaRect + boundary box points): https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
- cv2.findContours: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
- cv2.warpAffine + cv2.getRotationMatrix2D:  https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html

Connected Component Analysis (CCA): Character Segmentation & Filtering:
- https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/

Character Recognition (Template Matching):
- https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html
- https://www.freecodecamp.org/news/sort-dictionary-by-value-in-python/ (sorting dictionary by value)

Matplotlib / displaying results:
- Drawing a rectangle around a component: https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/ 
- Displaying multiple images in a matplotlib subplot: https://www.geeksforgeeks.org/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/


### `write_data.py`
Creating an XML file, adding child elements to parent items: https://docs.python.org/3/library/xml.etree.elementtree.html


### `parse_data.py`
Parsing/reading XML file using xml.etree: https://docs.python.org/3/library/xml.etree.elementtree.html