# f20pa-anpr

Python and OpenCV solution to implement a standard ANPR solution, consisting of pipeline stages: image pre-processing, 
number plate extraction, character segmentation and character recognition.
predicted match. The main investigatory aim of this project is to assess the effectiveness of image
pre-processing Techniques in improving ANPR reading accuracy

4th Year Undergraduate BSc (Hons) Dissertation Project. John McIver. Heriot-Watt University, Edinburgh. School of
Mathematical and Computer Sciences. Department of Computer Science. jm2006@hw.ac.uk

All rights reserved. © 2024 John McIver.
Use of this project is allowed, granted that proper reference and credit is provided to the author.


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
              against the dataset. 1a :- Noise Removal (Bilateral Filtering)
              
              1b :- Improving Contrast (Adaptive Histogram Equalisation) 1c :-
              Noise Removal (Adaptive Gaussian Thresholding) (on) | Default: Otsu's
              Thresholding 1d :- Tilt Correction
              
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



## Code References
In-code citations are also listed above in function headers.

### `anpr.py`


### `parse_data.py`


### `write_data.py`