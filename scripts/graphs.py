import xml.etree.ElementTree as ET

"""
    Read only script which parses output XML files to generate results dataset and matplotlib graphs
"""


def parse_xml(file_path):
    print("+----------------+\n| Results Output |\n+----------------+\n Input File: ", file_path, "\n")
    tree = ET.parse(file_path)
    root = tree.getroot()

    # accuracy
    correct = 0
    limit = 12000

    total_processing_time = 0

    i = 0
    for item in root:
        # print("-----------------")
        if item.find("is_correct").text == "True":
            correct += 1
        total_processing_time += float(item.find("process_time_sec").text)
        # for child in item:
        # print(child.tag, child.text)
        # print("-----------------\n")

    accuracy = (correct / limit) * 100
    print("Accuracy: {}%".format(accuracy))
    print("Total processing time (sec): {}".format(total_processing_time))
    print("Avg processing time per input (sec) : {}".format(total_processing_time / limit))


parse_xml("/Users/jmciver/PycharmProjects/f20pa-anpr/xml_files/1a_1b_1c_1d_whiteplate_12000.xml")
