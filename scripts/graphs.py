import xml.etree.ElementTree as ET


def convert_seconds_to_miliseconds(sec):
    return  # todo


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
    print("+----------------+\n| Results Output |\n+----------------+\n Input File: ", file_path, "\n")
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
                "Error: Exception occured. Input file is either not an XML document, or the internal hierarchy is "
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

# parse_xml("/Users/jmciver/PycharmProjects/f20pa-anpr/xml_files/TEST_TEST2.xml")
