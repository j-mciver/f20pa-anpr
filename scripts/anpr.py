import argparse
import sys

import cv2
import os
import time
from matplotlib import pyplot as plt
import numpy as np

from scripts.parse_data import parse_xml, calc_group_tilt_degree_by_accuracy
from write_data import write_to_xml_file, store_results, clear_results

templates_dir = "./templates"
templates_list = sorted(os.listdir(templates_dir))

""" Convert an input RGB image to greyscale
    
    Attributes
    -----------
        - img : Input RGB image to be converted to greyscale
    
    Reference Usage: OpenCV Converting RGB images to Greyscale
        https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
"""


def convert_rgb_to_greyscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_bilateral_filter(img):
    return cv2.bilateralFilter(img, 7, 75, 75)


# reference: bilateral filter algorithm https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
def iterative_bilateral_filter(img):
    return apply_bilateral_filter(img)


""" Adaptive Histogram Equalisation
    - Improves the contrast of input image by locally examining regions of pixels, called neighbours, and distributes
    illumination/contrast by applying weighted value based on examined pixels. Assess the two peaks (low and
    high contrast).
    
    Reference: 
        https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/
"""


def adaptive_histogram_equalisation(img):
    ahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    return ahe.apply(img)


def adaptive_threshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)


def tilt_correction(th_img, component_mask):
    # reference: rotated rectangle: https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

    # 1 = cv2.RETR_EXTERNAL (exclude nested/internal contours)
    # 2 = cv2.CHAIN_APPROX_SIMPLE we want diagonal lines
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    number_plate = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(number_plate)
    box = np.intp(box)

    centre = number_plate[0]
    (x, y) = number_plate[1]
    angle = number_plate[2]

    output = cv2.cvtColor(th_img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(output, [box], 0, (0, 255, 0), 2)
    data_dict["uncorrected_tilt"] = output

    if angle > 45:  # force warpAffine to rotate clockwise
        matrix = cv2.getRotationMatrix2D(centre, angle + 270, 1)
        angle = 360 - (angle + 270)
    else:
        matrix = cv2.getRotationMatrix2D(centre, angle, 1)
    data_dict["angle"] = angle

    (h, w) = output.shape[:2]
    corrected_img = cv2.warpAffine(th_img, matrix, (w, h), flags=cv2.INTER_LINEAR)
    data_dict["corrected_img"] = corrected_img

    # print("TILT CORRECTION:")
    # print(centre)
    # print("x, y: ", (x, y))
    # print("angle of rotation (deg): ", angle)

    return corrected_img


def character_segmentation(th_img, s_1d):
    # Reference: Connected-Component Analysis (https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/)
    output = cv2.connectedComponentsWithStats(th_img, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    characters = []
    rect_border = []

    # sort components to appear based x-axis order (sorted on character, left to right)
    x_axis_sorted_components = list()
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        area = stats[i, cv2.CC_STAT_AREA]
        x_axis_sorted_components.append([x, i, area])

    list.sort(x_axis_sorted_components)

    # extract number plate from input image based on largest area
    max = 0
    idx = None
    for comp in x_axis_sorted_components:
        if comp[2] > max:
            max = comp[2]
            idx = comp[1]

    # check area and index
    if idx is not None and max > 0:
        component_mask = (labels == idx).astype("uint8") * 255
        corrected_img = tilt_correction(th_img, component_mask)

        if not s_1d:
            corrected_img = th_img
            data_dict["uncorrected_tilt"] = corrected_img
            data_dict["corrected_img"] = corrected_img

        output = cv2.connectedComponentsWithStats(corrected_img, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        char_img = np.zeros(corrected_img.shape, dtype="uint8")

        # sort components to appear based x-axis order (sorted on character, left to right)
        x_axis_sorted_components = list()
        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            x_axis_sorted_components.append([x, i])

        list.sort(x_axis_sorted_components)

        for i, j in x_axis_sorted_components:
            text = "component {}/{}".format(j + 1, numLabels)

            # print("[INFO] {}".format(text))

            x = stats[j, cv2.CC_STAT_LEFT]
            y = stats[j, cv2.CC_STAT_TOP]
            w = stats[j, cv2.CC_STAT_WIDTH]
            h = stats[j, cv2.CC_STAT_HEIGHT]
            area = stats[j, cv2.CC_STAT_AREA]
            # print("LABEL {}/{} stats: w = {}, h = {}, area = {}".format(j + 1, numLabels, w, h, area))

            # filter connected components by width, height and area of pixels
            if all((5 < w < 50, 40 < h < 65, 290 < area < 1200)):
                # output = cv2.cvtColor(corrected_img, cv2.COLOR_GRAY2RGB)
                # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # print("Keeping component {}".format(text))
                component_mask = (labels == j).astype("uint8") * 255
                characters.append(component_mask)
                rect_border.append([x, y, w, h])
                char_img = cv2.bitwise_or(char_img, component_mask)

                # cv2.imshow("Output", output)
                # cv2.waitKey(0)

    else:
        raise Exception("Error: Plate not found.")

    # domain knowledge (remove the distinguishing sign - will leafmost by X coordinate)
    # todo: ASSUMPTION - plates are in standard format that consists of 7 characters/digits only. dateless/private NPs
    # not included within project scope
    while len(characters) > 7:
        rect_border.pop(0)
        characters.pop(0)

    return char_img, rect_border, characters


def extract_characters(char_img, rect_border):
    extracted_char_templates = []

    # normalise extracted characters to be suitable for template matching
    for r in rect_border:
        # y:y + h, x:x + w (rows, columns)
        ext_char = char_img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
        ext_char = cv2.bitwise_not(ext_char)

        # resize to template to suitable width and height (30, 60)
        ext_char = cv2.resize(ext_char, (30, 60), cv2.INTER_LINEAR)

        # ext_char = cv2.cvtColor(ext_char, cv2.COLOR_RGB2GRAY)
        extracted_char_templates.append(ext_char)

    return extracted_char_templates


# https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html
def template_match(extracted_char_templates):
    confidence = []
    confidence_distribution = []
    max_confidence = 0
    best_guess_char = ""
    reg = ""

    for ext_char in extracted_char_templates:
        # cv2.imshow("ext_char", ext_char)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        tmp_dict = {}
        for template in templates_list:
            template_path = templates_dir + "/" + template

            # convert both images to greyscale for inputs to matchTemplate()
            tmp_img = cv2.imread(template_path)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGRA2GRAY)

            res = cv2.matchTemplate(ext_char, tmp_img, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            tmp_dict[template[0]] = max_val

            if max_val > max_confidence:
                max_confidence = max_val
                best_guess_char = template[0]

        print("MAX CONFIDENCE: ", max_confidence, " CHAR = ", best_guess_char)
        # todo: EXTREMELY HIGH DOMAIN LEVEL KNOWLEDGE ASSUMPTION - testing has shown that very poor confidence is most likely a 1 due to issues with extracting and comparing one as a char
        if max_confidence < 0.30:
            best_guess_char = '1'
        reg += best_guess_char
        confidence.append(max_confidence)

        tmp_dict = dict(reversed(sorted(tmp_dict.items(), key=lambda con: con[1])))
        confidence_distribution.append(tmp_dict)

        max_confidence = 0
        best_guess_char = ""

    for entry in confidence_distribution:
        print(entry)
    print(reg.upper())
    return reg, confidence, confidence_distribution


""" start() method - accepts input image directory and runs ANPR pipeline stages, returning predicted match 

    Toggleable Pre-processing Pipeline Stages:
        1a :- Noise Removal (Bilateral Filtering)
        1b :- Improving Contrast (Adaptive Histogram Equalisation)
        1c :- Noise Removal (Adaptive Gaussian Thresholding) (on) | Default: Otsu's Thresholding (off)
        1d :- Tilt Correction (Bilateral Transformation)"""


def start(image_list, image_dir, limit, s_1a, s_1b, s_1c, s_1d, plot_results, file_name=""):
    global data_dict
    data_dict = dict()
    clear_results()

    limit = limit
    count = 0
    for file in image_list:
        start_time = time.time()
        if file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg"):
            print("Processing {0}".format(file))
            image_path = image_dir + "/" + file
            image = cv2.imread(image_path)
            greyscale_img = convert_rgb_to_greyscale(image)
            # apply iterative bilateral filter
            if s_1a:
                filtered_image = iterative_bilateral_filter(greyscale_img)
            else:
                filtered_image = greyscale_img

            # adaptative histogram equalisation
            # AHE reference: https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/
            if s_1b:
                ahe_img = adaptive_histogram_equalisation(filtered_image)
            else:
                ahe_img = filtered_image

            # applying adaptive thresholding https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
            if s_1c:
                th_img = adaptive_threshold(ahe_img)
            else:
                th_val, th_img = cv2.threshold(ahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contrast_before_preprocessing = greyscale_img.std()
            contrast_after_preprocessing = ahe_img.std()

            # todo: assumption brightness categories thresholds are manually adjusted
            brightness = greyscale_img.mean()
            if brightness > 80:
                brightness_category = "bright"
            elif brightness < 50:
                brightness_category = "dark"
            else:
                brightness_category = "normal"

            # Character Segmentation
            th_img = cv2.bitwise_not(th_img)  # invert binary img OpenCV CCA expects black background, white foreground
            char_img, rect_border, characters = character_segmentation(th_img, s_1d)

            # Extract Characters from Original Input Image
            ext_char_templates = extract_characters(char_img, rect_border)

            # Template Matching
            reg, confidence, confidence_distribution = template_match(ext_char_templates)

            # Number Plate Assumption: (A1) The letter 'I'/'i' does not appear in NPs, only 1. (REF Gov Standards)
            # A2: The letter 'O' and number '0' are equivalent (REF Gov Standard)
            # Equivalent Character Assumptions (todo: domain knowledge assumption)
            if "I" in file:
                file = file.replace("I", "1")
            if "O" in file:
                file = file.replace("O", "0")

            process_time = time.time() - start_time
            print("%s took %s seconds\n" % (file, process_time))

            # Contents of reg are updated, need to preserve old state for writing result to XML file
            tmp_reg = reg.upper()
            if reg.upper() != file[:7]:
                actual_reg = file[:7].lower()
                # find the index(es) that were incorrect, update confidence array to reflect true prediction
                for i in range(min(len(reg), len(actual_reg))):
                    # predicted does not equal actual character
                    if reg[i] != actual_reg[i]:
                        confidence[i] = confidence_distribution[i].get(actual_reg[i])
                        reg = reg[:i] + actual_reg[i] + reg[i + 1:]

            if reg == "":
                print('failed to detect chars')

            psnr = cv2.PSNR(greyscale_img, th_img)

            # --- store data for write_data.py  ---
            store_results([
                file[:7],
                tmp_reg,
                tmp_reg == file[:7],
                process_time,
                confidence,
                confidence_distribution,
                psnr,
                data_dict["angle"],
                contrast_before_preprocessing,
                contrast_after_preprocessing,
                brightness,
                brightness_category
            ])

            """ --- DISPLAY PROCESSED IMAGES --- 
                Contents are only displayed if -p command line arg is provided (plot results enabled)
            """
            plot_results = plot_results
            if plot_results:
                rows = 4
                cols = 4

                # OpenCV reads images BGR, matplotlib reads in RGB. Convert all images.
                image = convert_bgr_rgb(image)
                greyscale_img = convert_bgr_rgb(greyscale_img)
                filtered_image = convert_bgr_rgb(filtered_image)
                ahe_img = convert_bgr_rgb(ahe_img)
                th_img = convert_bgr_rgb(th_img)
                uncorrected_tilt_img = convert_bgr_rgb(data_dict["uncorrected_tilt"])
                corrected_tilt_img = convert_bgr_rgb(data_dict["corrected_img"])
                cca_output = corrected_tilt_img.copy()

                # reference: drawing around component border
                # https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
                for r in rect_border:
                    # (x, y), (x + w, y + h)
                    cv2.rectangle(cca_output, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 255, 0), 3)

                # Reference displaying multiple images in matplotlib subplots:
                # https://www.geeksforgeeks.org/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/
                fig = plt.figure(figsize=(16, 8))
                display_results(fig, 3, 3, 2, [[image, "Input Image " + file]])

                # noise removal stage data
                noise_removal_data = [[greyscale_img, "Greyscaled Input RGB Image"],
                                      [filtered_image, "Bilateral Filtered Image"],
                                      [ahe_img, "Adaptive Histogram Equalisation"],
                                      [th_img, "Adaptive Gaussian Thresholding"]]

                display_results(fig, rows, cols, 5, noise_removal_data)

                # tilt correction data
                display_results(fig, rows, cols, 10, [[uncorrected_tilt_img, "Uncorrected Tilt"]])
                props = dict(boxstyle='round', facecolor='wheat')
                angle_str = "{:0.2f}° tilt".format(data_dict["angle"])
                plt.text(0.5, 170, angle_str, fontsize=14, verticalalignment='top', bbox=props)
                display_results(fig, rows, cols, 11, [[corrected_tilt_img, "Corrected Tilt"]])
                display_results(fig, rows, cols, 14, [[cca_output, "Connected Component Analysis (CCA)"]])

                # output stage data
                # NP registration prediction / match:
                match_strength = ""
                for con in confidence:
                    match_strength = match_strength + str(con)[:5] + ", "

                plt.text(850, 100, reg.upper().replace("", "  "), fontsize="40", fontname='Arial', color="black")
                plt.text(900, 150, match_strength, fontsize="15", fontname='Arial', color="black")

                plt.subplots_adjust(hspace=1.1)
                plt.show()

            count = count + 1
            if count == limit:
                # Write analytical metrics to XML file
                xml_dir = "/Users/jmciver/PycharmProjects/f20pa-anpr/xml_files/"
                file = file_name + "t4_v10_test.xml"
                write_to_xml_file(file)
                parse_xml(xml_dir + file)
                break


def display_results(fig, rows, cols, index, data):
    i = index
    for img, title in data:
        fig.add_subplot(rows, cols, i)
        plt.imshow(img, aspect='auto')
        plt.title(title, {'fontname': 'Arial'})
        plt.axis("off")
        i = i + 1


def convert_bgr_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def parse_stage_args(stages):
    stages = stages.replace(" ", "").strip()

    # no pre-processing pipeline stages specified, all are toggled off.
    if len(stages) == 0 or stages == "" or stages is None:
        return []
    elif len(stages) == 2:
        return [stages]
    elif len(stages) == 4:
        s1 = stages[:2]
        s2 = stages[2:4]
        return [s1, s2]
    elif len(stages) == 6:
        s1 = stages[:2]
        s2 = stages[2:4]
        s3 = stages[4:6]
        return [s1, s2, s3]
    elif len(stages) == 8:
        s1 = stages[:2]
        s2 = stages[2:4]
        s3 = stages[4:6]
        s4 = stages[6:8]
        return [s1, s2, s3, s4]

    else:
        raise Exception("Error: Invalid stage, or number of stages, specified.")


def call_preprocessing_pipeline(stages):
    s_1a = False
    s_1b = False
    s_1c = False
    s_1d = False
    for stage in stages:
        if stage == "1a":
            s_1a = True
        elif stage == "1b":
            s_1b = True
        elif stage == "1c":
            s_1c = True
        elif stage == "1d":
            s_1d = True
    return s_1a, s_1b, s_1c, s_1d


# reference: argparse usage https://docs.python.org/3/library/argparse.html
def cl_args_handler():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', type=str, help="The directory containing the input image dataset.", required=True)
        parser.add_argument('-l', type=int, help="The number of files which should be processed from the input image "
                                                 "dataset. Unspecified limit will default to processing all available "
                                                 "images from the input directory.", required=False)
        parser.add_argument('-s', type=str,
                            help="Specify the pre-processing pipeline stages which should be run against the dataset."
                                 "\n1a :- Noise Removal (Bilateral Filtering)"
                                 "\n1b :- Improving Contrast (Adaptive Histogram Equalisation)"
                                 "\n1c :- Noise Removal (Adaptive Thresholding) (on) | Default: Otsu's Thresholding"
                                 "\n1d :- Tilt Correction (Bilateral Transformation)", required=True)
        parser.add_argument('-p', type=str, help="Plot and display the results of each pipeline processing"
                                                 " stage. By default this will write result data to a directory",
                            required=False)
        args = parser.parse_args()

        if os.path.exists(args.d):
            image_dir = args.d
            image_list = sorted(os.listdir(args.d))
            print("Valid path: ", args.d)
        else:
            raise Exception("Error: Invalid directory path provided.")

        if args.l is None:
            limit = len(image_list)
        elif isinstance(args.l, int) and args.l <= len(image_list) and len(image_list) > 0:
            limit = args.l
            print("Limit :- ", limit)
        else:
            raise Exception("Error: Limit out of bounds. Limit entered exceeds the amount of items present in the "
                            "directory.")

        if args.p is not None:
            if args.p.lower() == "true":
                plot_results = True
            elif args.p.lower() == "false":
                plot_results = False
        else:
            plot_results = False

        print("Plot Results: ", plot_results)

        stages = parse_stage_args(args.s)

        if stages == [] and len(stages) == 0:
            # no stages specified, call start with not enabled on all
            start(image_list, image_dir, limit, False, False, False, False, plot_results)
        elif len(stages) > 0:
            s_1a, s_1b, s_1c, s_1d = call_preprocessing_pipeline(stages)
            start(image_list, image_dir, limit, s_1a, s_1b, s_1c, s_1d, plot_results)

        print("\nSTAGES enabled: ", stages)


stage_permutations = [
    ["1a", "1b", "1c", "1d"],
    ["1a", "1b", "1c"],
    ["1a", "1b", "1d"],
    ["1a", "1b"],
    ["1a", "1c", "1d"],
    ["1a", "1c"],
    ["1a", "1d"],
    ["1a"],
    ["1b", "1c", "1d"],
    ["1b", "1c"],
    ["1b", "1d"],
    ["1b"],
    ["1c", "1d"],
    ["1c"],
    ["1d"],
    [""]
]


def iter_stage_permutations(stage_perms):
    for perm in stage_perms:
        file_name = "_".join(perm) + "_"
        s_1a, s_1b, s_1c, s_1d = call_preprocessing_pipeline(perm)
        image_dir = "/Users/jmciver/Documents/Y4S1/F20PA/DISSERTATION-MATERIAL/UKLicencePlateDataset/yellowplate_augmented"
        image_list = sorted(os.listdir(
            "/Users/jmciver/Documents/Y4S1/F20PA/DISSERTATION-MATERIAL/UKLicencePlateDataset/yellowplate_augmented"))
        start(image_list, image_dir, 12000, s_1a, s_1b, s_1c, s_1d, False, file_name)


# iter_stage_permutations(stage_permutations)


def parse_xml_files(dir):
    xml_list = sorted(os.listdir(dir))
    for xml_file in xml_list:
        if xml_file.endswith(".xml"):
            print("\n", xml_file)
            output = parse_xml(dir + "/" + xml_file)
            # print(output)
            # with open(xml_file[:-4] + "_output_results.txt", "w") as res:
            #     res.write(output)


# calc_group_tilt_degree_by_accuracy(
#     "/Users/jmciver/PycharmProjects/f20pa-anpr/xml_files/tc_enabled_tilt_variation_data.xml")

# parse_xml_files("/Users/jmciver/PycharmProjects/f20pa-anpr/xml_files/v2_yellowplate_safe_store")

cl_args_handler()
#
# REFERENCES
# argparse usage https://docs.python.org/3/library/argparse.html
# https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
# Bilateral Filter: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
# Peak-Signal to Noise Ratio: (OpenCV) https://shimat.github.io/opencvsharp_docs/html/23f56d6b-49ef-3365-5139-e75712c20fe4.htm
# otsu's method + adaptive thresholding: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
# adaptive historgram equalisation https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/
# cca (identifying NP and characters in input image) https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
# https://www.geeksforgeeks.org/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/
# https://learnopencv.com/cropping-an-image-using-opencv/#cropping-using-opencv
# resizing an image (opencv) tutorial reference: https://learnopencv.com/image-resizing-with-opencv/
# https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html (tilt correction based on affine transformation)
# plot_results placing text boxes https://matplotlib.org/3.3.4/gallery/recipes/placing_text_boxes.html
# sorting python dictionary by values: https://www.freecodecamp.org/news/sort-dictionary-by-value-in-python/
