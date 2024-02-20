import xml.etree.ElementTree as ET

"""
    Read only script which parses output XML files to generate results dataset and matplotlib graphs
"""
def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    i = 0
    for item in root:
        print("-----------------")
        if i >= 5:
            return
        for child in item:
            print(child.tag, child.text)
        print("-----------------\n")
        i += 1


parse_xml("/Users/jmciver/PycharmProjects/f20pa-anpr/xml_files/1a_1b_1c_1d_whiteplate_12000.xml")