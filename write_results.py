import csv

""""
<registration_text> AA10QYN </registration_text>
<predicted_text> AA10QYN </predicted_text>
<is_correct>true</is_correct>
<processing_time_sec> 0.002183sec </processing_time_sec>

<max_confidence>
    <1st_char>90</1st_char>
    <2nd_char>80</2nd_char>
            ...
</max_confidence>

<confidence_distribution>
    <1>a:90,c:90,d:89,84..</1st_char>
    <2>a:90,z:29,c:8 ..</2nd>
    <3>
    + 0-9 digits
    .. 7 items * 35 matches in dictionary
     .. 7 items in every tag (for 7 characters in plate) show distribution for each predicted match
</confidence_distribution>

psnr
degree_of_tilt
contrast_before_preprocessing
contrast_after_preprocessing
<brightness> float </brightness>
brightness_category> bright/normal/dark
"""

global data
data = []


def store_results(res):
    data.append(res)


def parse_data(data):
    return

def write_to_xml_file():
    parse_data(data)
    for d in data:
        print(d)
    return

# References:
# Creating an XMl file using xml.etree: https://docs.python.org/3/library/xml.etree.elementtree.html
