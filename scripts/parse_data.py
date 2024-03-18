import xml.etree.ElementTree as ET

import numpy as np


def psnr_range(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    list_psnr = []
    for item in root:
        try:
            list_psnr.append(float(item.find("psnr").text))
        except:
            raise Exception(
                "Error: Exception occured. Input file is either not an XML document, or the internal hierarchy is "
                "invalid.")
    list_psnr = sorted(list_psnr)
    psnr = np.array(list_psnr)
    print("sd ", psnr.std())
    print("avg ", psnr.mean())
    print("range ", list_psnr[-1], " to ", list_psnr[0])


def avg_contrast_before_and_after(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    total = 0
    avg_contrast_before = 0
    avg_contrast_after = 0

    for item in root:
        try:
            if item.find("registration_text").text == "AB05WKZ":
                print(item.find("registration_text").text)
                print("before sd abo5wkz ", str(item.find("contrast_before_preprocessing").text))
                print("after sd ", str(item.find("contrast_after_preprocessing").text))
            avg_contrast_before += float(item.find("contrast_before_preprocessing").text)
            avg_contrast_after += float(item.find("contrast_after_preprocessing").text)
            total += 1

        except:
            raise Exception(
                "Error: Exception occured. Input file is either not an XML document, or the internal hierarchy is "
                "invalid.")

    avg_contrast_after /= 12000
    avg_contrast_before /= 12000
    print("avg contrast after", avg_contrast_after)
    print('avg contrast before', avg_contrast_before)
    if total != 12000:
        raise Exception("Error: Invalid Total.")


"""" Calculates bins for each category of contrast (brightness) by errors. 
     Example: 100 errors: 75 low-brightness, 15 normal, 10 bright."""


def composition_of_errors_by_contrast(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    num_errors = 0
    low = 0
    normal = 0
    bright = 0

    for item in root:
        try:
            if item.find("is_correct").text == "False":
                num_errors += 1
                brightness = item.find("brightness_category").text
                if brightness == "dark":
                    low += 1
                elif brightness == "normal":
                    normal += 1
                elif brightness == "bright":
                    bright += 1


        except:
            raise Exception(
                "Error: Exception occured. Input file is either not an XML document, or the internal hierarchy is "
                "invalid.")

    print("Number of Errors: " + str(num_errors))
    print("Composition: low -", low, " | normal - ", normal, " | bright - ", bright)


""" Calculates the reading accuracy of a match, grouped by degree of tilt.
    Returns: (tilt degree, accuracy) """


def calc_group_tilt_degree_by_accuracy(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    res = []

    for item in root:
        try:
            degree_of_tilt = item.find("degree_of_tilt").text
            is_correct = item.find("is_correct").text
            if is_correct == "True":
                res.append((float(degree_of_tilt), 100))
            else:
                num_chars_correct = 7
                reg = item.find("predicted_text").text
                if reg is None:
                    reg = ""
                actual_reg = item.find("registration_text").text

                # calculate the accuracy of the guess (how many chars were predicted correctly)
                if len(reg) != 7:
                    diff = len(actual_reg) - len(reg)
                    num_chars_correct -= diff

                for i in range(min(len(reg), len(actual_reg))):
                    # predicted does not equal actual character
                    if reg[i] != actual_reg[i]:
                        num_chars_correct -= 1

                res.append((float(degree_of_tilt), (num_chars_correct / 7) * 100))

        except:
            raise Exception(
                "Error: Exception occured. Input file is either not an XML document, or the internal hierarchy is "
                "invalid.")
    for deg, acc in res:
        print('(' + str(deg) + "," + str(acc) + ")")


def calc_misread_chars_dict(incorrect_reg):
    misread_chars_dict = {}
    # todo: ASSUMPTION plates will only consist of 7 characters (UK standard, does not include private/dateless)
    for reg in incorrect_reg:
        for i in range(min(len(reg[0]), len(reg[1]))):
            # predicted does not equal actual character
            if reg[0][i] != reg[1][i]:
                # update count if key exists
                if (reg[1][i], reg[0][i]) in misread_chars_dict:
                    misread_chars_dict[(reg[1][i], reg[0][i])] += 1
                else:
                    misread_chars_dict[(reg[1][i], reg[0][i])] = 1
    return misread_chars_dict


""" Calculates the mean confidence for each template match.
    Calculates mean confidence on both positive and negative results, where the prediction was correct or incorrect.
    Uses confidence[] list which stores the top match confidence result for every extracted character.

    Returns: dictionary containing character/digit mapped to mean confidence """


def calc_mean_confidence(dict, reg, confidence):
    if (len(reg) != len(confidence)):
        raise Exception("Error: Numbers of letters/digits does not match confidence array values.")

    for i in range(0, len(reg)):
        if reg[i] in dict:
            dict[reg[i]][1] += 1
            dict[reg[i]][0] += confidence[i]
        else:
            dict[reg[i]] = [confidence[i], 1]


"""
    Read only script which parses output XML files to generate results dataset
"""


def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    correct = 0
    incorrect_reg = []
    limit = 0
    total_processing_time = 0
    psnr = 0

    mean_confidence_dict = dict()

    for item in root:
        limit += 1
        try:
            # reading accuracy
            if item.find("is_correct").text == "True":
                correct += 1
            else:
                incorrect_reg.append([item.find("predicted_text").text, item.find("registration_text").text])

            # processing time / avg processing time
            total_processing_time += float(item.find("process_time_sec").text)

            # avg psnr
            psnr += float(item.find("psnr").text)

            # mean confidence
            reg = item.find("predicted_text").text
            max_confidence = []
            for con in item.find("max_confidence"):
                max_confidence.append(float(con.text))
            calc_mean_confidence(mean_confidence_dict, reg, max_confidence)


        except:
            raise Exception(
                "Error: Exception occurred. Input file is either not an XML document, or the internal hierarchy is "
                "invalid.")

    for key, arr in mean_confidence_dict.items():
        mean_confidence_dict[key] = arr[0] / arr[1]
    mean_confidence_dict = sorted(mean_confidence_dict.items(), key=lambda x: x[1])

    misread_chars = calc_misread_chars_dict(incorrect_reg)
    accuracy = (correct / limit) * 100

    output_str = """+----------------+\n| Results Output |\n+----------------+\n Input File: {} \n
    Accuracy: {:0.5f}%
    Total processing time (sec): {:0.5f}
    Avg processing time per input (sec) : {:0.5f}
    Error rate: {:0.5f}% | {}/{}
    Incorrect Registrations (Predicted, Expected): {}
    Most Commonly Incorrect Characters (Actual character was misread as ... N times) | {}
    Average PSNR (dB) {:0.5f}
    Mean Confidence per Character {}
    """.format(file_path,
               accuracy,
               total_processing_time,
               total_processing_time / limit,
               (len(incorrect_reg) / limit) * 100,
               len(incorrect_reg),
               limit,
               incorrect_reg,
               misread_chars,
               psnr / limit,
               mean_confidence_dict)

    print(output_str)
    return output_str

# parse_xml("/Users/jmciver/PycharmProjects/f20pa-anpr/xml_files/v2_whiteplate_safe_store/1a__v2.xml")
psnr_range("/Users/jmciver/PycharmProjects/f20pa-anpr/xml_files/v2_whiteplate_safe_store/1a_1c__v2.xml")