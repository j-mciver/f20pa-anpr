import xml.etree.ElementTree as ET

global data
data = []

xml_dir = "../xml_files/"


def store_results(res):
    data.append(res)

def clear_results():
    data.clear()


def create_child_elements(data, parent):
    i = 1
    for entry in data:
        child_tag = "c" + str(i)
        ET.SubElement(parent, child_tag).text = str(entry)
        i += 1


"""" Writes data captures from anpr.py to XML file 

        Example:
        <registration_text>AA10QYN</registration_text>
        <predicted_text>AA10QYN</predicted_text>
        <is_correct>True/False</is_correct>
        <processing_time_sec>0.002183</processing_time_sec>
    
        <max_confidence>
            <c1>0.90</c1>
            <c2>0.80</c2>
                ...
        </max_confidence>
    
        <confidence_distribution>
            <c1>a:90,c:90,d:89,84..</c1>
            <c2>a:90,z:29,c:8 ..</c2>
            .. 7 items * 34 templates in dictionary
        </confidence_distribution>
        <psnr>7.613</psnr>
        <degree_of_tilt>3.6</degree_of_tilt>
        <contrast_before_preprocessing>30</contrast_before_preprocessing>
        <contrast_after_preprocessing>90</contrast_after_preprocessing>
        <brightness>155</brightness>
        <brightness_category> bright/normal/dark</brightness_category>
        
        
        References:
        Creating an XMl file using xml.etree: https://docs.python.org/3/library/xml.etree.elementtree.html
"""
def write_to_xml_file(file_name):
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
    tree.write(xml_dir + file_name)
