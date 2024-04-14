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

Output:
```
/usr/bin/python3 /Users/jmciver/PycharmProjects/f20pa-anpr/scripts/parse_data.py 
+----------------+
| Results Output |
+----------------+
 Input File: /Users/jmciver/PycharmProjects/f20pa-anpr/xml_files/v2_whiteplate_safe_store/1a_1b_1c_1d__v2.xml 

    Accuracy: 98.91667%
    Total processing time (sec): 243.28439
    Avg processing time per input (sec) : 0.02027
    Error rate: 1.08333% | 130/12000
    Incorrect Registrations (Predicted, Expected): [['AT33BC1', 'AT33HC1'], ['AD03SCQ', 'AU03SCQ'], ['AD50FZP', 'AU50FZP'], ['BC98DUB', 'BC98DUH'], ['F53V1BN', 'BF53WBN'], ['B187BJB', 'B187HJB'], ['BJ82WJB', 'BJ82WJH'], ['BN38WDD', 'BN38WDU'], ['BP25DHY', 'BP25UHY'], ['CB10PDN', 'CB10PUN'], ['CC05VQD', 'CC05VQU'], ['CB05DR1', 'CH05DR1'], ['CL98GVB', 'CL98GVH'], ['CX78B0V', 'CM78B0V'], ['CSD4WKX', 'CS04WKX'], ['CW60BLL', 'CW60HLL'], ['DA35RDU', 'DA35RUU'], ['DB65F0R', 'DH65F0R'], ['DL77BKJ', 'DL77HKJ'], ['DV73BYM', 'DV73HYM'], ['DX86YCD', 'DX86YCU'], ['EE48DQN', 'EE48UQN'], ['EF57QDA', 'EF57QUA'], ['EM26DYF', 'EM26UYF'], ['EQ3BVKB', 'EQ38VKB'], ['EV49DXT', 'EV49UXT'], ['EY21EVX', 'EY24EVX'], ['EY76BWG', 'EY76HWG'], ['EZD9G0C', 'EZ09G0C'], ['F143XHA', 'F143KHA'], ['F195TBD', 'F195TBU'], ['FV28BER', 'FV28HER'], ['G879WXE', 'GH79WXE'], ['G130KDB', 'G130KDH'], ['HB53DWJ', 'HB53UWJ'], ['BE05CZF', 'HE05CZF'], ['HK38DYB', 'HK38DYH'], ['BQ40XEX', 'HQ40KEX'], ['BR59JBB', 'HR59JBB'], ['BS49MZR', 'HS49MZR'], ['HU34XAC', 'HU34KAC'], ['HW01BXD', 'HW01BKD'], ['1B17GBV', '1B17GHV'], ['1E52BAK', '1E52HAK'], ['1V93XZX', '1V93KZX'], ['1X65HDM', '1X65HUM'], ['JD84DCL', 'JD84UCL'], ['JK52KMB', 'JK52KMH'], ['JP72PBB', 'JP72PHB'], ['JW73MBG', 'JW73MHG'], ['JY53DLW', 'JY53ULW'], ['KB26DTY', 'KB26UTY'], ['LD07PAD', 'LD07PAU'], ['LJ03BH1', 'LJ03HH1'], ['LQ14SKG', 'LQ44SKG'], ['LQ86DHN', 'LQ86UHN'], ['LU42DZF', 'LU42UZF'], ['LD84ENV', 'LU84ENV'], ['XA20ASY', 'MA20ASY'], ['ME68BAG', 'ME68HAG'], ['MD48PMB', 'MU48PMB'], ['MV84BSK', 'MV84HSK'], ['NG28DCL', 'NG28UCL'], ['N116BGW', 'N116HGW'], ['069F17K', 'N069FWK'], ['NU01GJX', 'NU01GJM'], ['0F82DGN', '0F82UGN'], ['0K98DZG', '0K98UZG'], ['0043BDU', '0043BUU'], ['0R23XNT', '0R23KNT'], ['0U23BZJ', '0U23HZJ'], ['PA12UGB', 'PA12UGH'], ['PX94MZA', 'PK94MZA'], ['QD22QPB', 'QD22QPH'], ['QJ69B0F', 'QJ69H0F'], ['QK59GRD', 'QK59GRU'], ['QK95VDX', 'QK95VDK'], ['QT65YZ', 'QT65YZM'], ['QT92RXB', 'QT92RXH'], ['QY33SRD', 'QY33SRU'], ['RB071AB', 'RB071AH'], ['RC701BD', 'RC701HD'], ['RE06DQJ', 'RE06UQJ'], ['SB29EJ0', 'SH29EJ0'], ['SB45QYN', 'SH45QYN'], ['SB52XAZ', 'SH52XAZ'], ['SK63ARD', 'SK63ARU'], ['SP59WGW', 'SP59WGH'], ['TG59DVY', 'TG59UVY'], ['T146TS0', 'T146ES0'], ['TL03BNQ', 'TL03HNQ'], ['TP71DUM', 'TP71UUM'], ['UB21D1D', 'UB21U1D'], ['DC40RR0', 'UC40RR0'], ['UD77VVB', 'UD77VVH'], ['UJ17BN0', 'UJ17HN0'], ['DM11WTB', 'UM11WTB'], ['M55JL11', 'UM55JLW'], ['UP80ADN', 'UP80AUN'], ['UQ64FUB', 'UQ64FUH'], ['UQ64BUV', 'UQ64HUV'], ['DR73MT1', 'UR73MT1'], ['DS76JXQ', 'US76JXQ'], ['DW311YR', 'UW311YR'], ['DY44JHF', 'UY44JHF'], ['VA68SQB', 'VA68SQH'], ['VB02FBV', 'VB02FHV'], ['VB73QXM', 'VH73QXM'], ['V102BRX', 'V102HRX'], ['VT92DBU', 'VT92DHU'], ['4000FP1', 'W000FP1'], ['1T08BVW', 'WT08HVW'], ['1W311AG', 'WW311AG'], ['Z84110J', 'WZ84W0J'], ['XF06DDX', 'XF06UDX'], ['XM10BQP', 'XM10HQP'], ['XR10BKY', 'XR10HKY'], ['XT90BGJ', 'XT90HGJ'], ['XV30AXB', 'XV30AKB'], ['X128SVH', 'XW28SVH'], ['XW66BHA', 'XW66HHA'], ['YF72ADV', 'YF72AUV'], ['YB47QPU', 'YH47QPU'], ['YN27YBU', 'YN27YHU'], ['Y057ADV', 'Y057AUV'], ['YP87AXA', 'YP87AKA'], ['YS51DYH', 'YS51UYH'], ['ZJ45JDP', 'ZJ45JUP'], ['ZK52DYV', 'ZK52UYV'], ['ZU50EBB', 'ZU50EBH']]
    Most Commonly Incorrect Characters (Actual character was misread as ... N times) | {('H', 'B'): 55, ('U', 'D'): 47, ('B', 'F'): 1, ('F', '5'): 1, ('5', '3'): 1, ('3', 'V'): 1, ('W', '1'): 6, ('M', 'X'): 3, ('0', 'D'): 2, ('8', 'B'): 1, ('4', '1'): 3, ('K', 'X'): 10, ('H', '8'): 1, ('N', '0'): 1, ('0', '6'): 1, ('6', '9'): 1, ('9', 'F'): 1, ('F', '1'): 1, ('W', '7'): 1, ('H', 'W'): 1, ('E', 'T'): 1, ('U', 'M'): 1, ('M', '5'): 1, ('5', 'J'): 1, ('J', 'L'): 1, ('L', '1'): 1, ('W', '4'): 1, ('W', 'Z'): 1, ('Z', '8'): 1, ('8', '4'): 1}
    Average PSNR (dB) 5.90086
    Mean Confidence per Character [('1', 0.6477586134258269), ('M', 0.6579845695797697), ('H', 0.6741987273890133), ('W', 0.6847730160297192), ('N', 0.6988147164890635), ('R', 0.7583133034475621), ('E', 0.7644832559769422), ('U', 0.768258626497593), ('D', 0.7785976577201971), ('K', 0.7819341072209511), ('Q', 0.7829034218967038), ('B', 0.7894491797968763), ('G', 0.7941400603186793), ('L', 0.7987281753221291), ('8', 0.809076929655797), ('2', 0.8107832444436622), ('9', 0.8155681885102145), ('0', 0.8161541259903101), ('6', 0.8218476654930824), ('5', 0.8231834144201936), ('F', 0.8344852047458479), ('S', 0.8384096262041842), ('Z', 0.840961279377969), ('3', 0.8446457346088493), ('C', 0.8484960031109127), ('J', 0.8501280821785702), ('X', 0.8574827002814945), ('P', 0.8583730029444743), ('A', 0.8751639688884707), ('V', 0.8778544515240039), ('4', 0.8817212020418828), ('T', 0.890083639154542), ('Y', 0.895320962508657), ('7', 0.8966437549865957)]
    
```

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