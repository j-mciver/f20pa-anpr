import xml.etree.ElementTree as ET

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
    <psnr>
    <degree_of_tilt>
    <contrast_before_preprocessing>
    <contrast_after_preprocessing>
    <brightness> float </brightness>
    <brightness_category> bright/normal/dark
"""

global data
data = []


def store_results(res):
    data.append(res)


def create_child_elements(data, parent):
    i = 1
    for entry in data:
        child_tag = "c" + str(i)
        ET.SubElement(parent, child_tag).text = str(entry)
        i += 1


def write_to_xml_file():
    root = ET.Element('root')
    for entry in data:
        item = ET.SubElement(root, 'item')

        registration_text = ET.SubElement(item, 'registration_text')
        registration_text.text = entry[0]

        predicted_text = ET.SubElement(item, 'predicted_text')
        predicted_text.text = entry[1]

        is_correct = ET.SubElement(item, 'is_correct')
        is_correct.text = str(entry[2])

        process_time_sec = ET.SubElement(item, 'process_time_sec')
        process_time_sec.text = str(entry[3])

        max_confidence = ET.SubElement(item, 'max_confidence')
        create_child_elements(entry[4], max_confidence)

        confidence_distribution = ET.SubElement(item, 'confidence_distribution')
        create_child_elements(entry[5], confidence_distribution)

        psnr = ET.SubElement(item, 'psnr')
        psnr.text = str(entry[6])

        degree_of_tilt = ET.SubElement(item, 'degree_of_tilt')
        degree_of_tilt.text = str(entry[7])

        contrast_before_preprocessing = ET.SubElement(item, 'contrast_before_preprocessing')
        contrast_before_preprocessing.text = str(entry[8])

        contrast_after_preprocessing = ET.SubElement(item, 'contrast_after_preprocessing')
        contrast_after_preprocessing.text = str(entry[9])

        brightness = ET.SubElement(item, 'brightness')
        brightness.text = str(entry[10])

        brightness_category = ET.SubElement(item, 'brightness_category')
        brightness_category.text = entry[11]

    tree = ET.ElementTree(root)
    # todo make UUID and cpature last date + time.now() + UUID [:-4]
    tree.write('xml_files/TEST.xml')

# References:
# Creating an XMl file using xml.etree: https://docs.python.org/3/library/xml.etree.elementtree.html
