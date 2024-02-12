import argparse
import sys

import cv2
import os
import time
from matplotlib import pyplot as plt
import numpy as np

# image_dir = "/Users/jmciver/Documents/Y4S1/F20PA/DISSERTATION-MATERIAL/UKLicencePlateDataset/whiteplate_augmented"
# image_dir = "/Users/jmciver/Documents/Y4S1/F20PA/DISSERTATION-MATERIAL/UKLicencePlateDataset/yellowplate_augmented"
# todo: image dir must not be hardcoded - make this an embedded project folder potentially on GitHub? + test yellow plates
# image_list = sorted(os.listdir(image_dir))

templates_dir = "/Users/jmciver/PycharmProjects/f20pa-anpr/templates"
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
    fimg = apply_bilateral_filter(img)
    psnr = cv2.PSNR(img, fimg)
    stack.append({"PSNR": psnr})
    print("PSNR: %s" % (psnr))
    return fimg


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

    if angle > 45:  # force warpAffine to rotate clockwise
        matrix = cv2.getRotationMatrix2D(centre, angle + 270, 1)
        angle = 360 - (angle + 270)
    else:
        matrix = cv2.getRotationMatrix2D(centre, angle, 1)
    stack.append(angle)

    (h, w) = output.shape[:2]
    corrected_img = cv2.warpAffine(th_img, matrix, (w, h), flags=cv2.INTER_LINEAR)

    print("TILT CORRECTION:")
    print(centre)
    print("x, y: ", (x, y))
    print("angle of rotation (deg): ", angle)

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
            print(comp[2])

    # check area and index
    if idx is not None and max > 0:
        component_mask = (labels == idx).astype("uint8") * 255
        if s_1d:
            corrected_img = tilt_correction(th_img, component_mask)
        else:
            corrected_img = th_img
        output = cv2.connectedComponentsWithStats(corrected_img, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        char_img = np.zeros(corrected_img.shape, dtype="uint8")

        # sort components to appear based x-axis order (sorted on character, left to right)
        x_axis_sorted_components = list()
        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            x_axis_sorted_components.append([x, i])

        list.sort(x_axis_sorted_components)
        print("TEST HELLO", x_axis_sorted_components)

        for i, j in x_axis_sorted_components:
            text = "component {}/{}".format(j + 1, numLabels)

            # print a status message update for the current connected component
            print("[INFO] {}".format(text))

            x = stats[j, cv2.CC_STAT_LEFT]
            y = stats[j, cv2.CC_STAT_TOP]
            w = stats[j, cv2.CC_STAT_WIDTH]
            h = stats[j, cv2.CC_STAT_HEIGHT]
            area = stats[j, cv2.CC_STAT_AREA]
            print("LABEL {}/{} stats: w = {}, h = {}, area = {}".format(j + 1, numLabels, w, h, area))

            # filter connected components by width, height and area of pixels
            if all((5 < w < 50, 40 < h < 65, 290 < area < 1200)):
                # output = cv2.cvtColor(corrected_img, cv2.COLOR_GRAY2RGB)
                # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)

                print("Keeping component {}".format(text))
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
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    max_confidence = 0
    best_guess_char = ""
    reg = ""

    threshold = 0.0
    for ext_char in extracted_char_templates:
        # cv2.imshow("ext_char", ext_char)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        for template in templates_list:
            template_path = templates_dir + "/" + template

            # convert both images to greyscale for inputs to matchTemplate()
            tmp_img = cv2.imread(template_path)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGRA2GRAY)

            res = cv2.matchTemplate(ext_char, tmp_img, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # todo: implement more intelligent logic against controlling confidence (will have to write assumption)
            # assumption is 2 character, 2 digits, 3 characters so 3rd and 4th itertaion picks highest correlation from number template set
            if max_val >= threshold:
                if max_val > max_confidence:
                    max_confidence = max_val
                    best_guess_char = template[0]
                # print("match found! confidence: ", max_val)
                # print("CHARACTER:",template[0])
        print("MAX CONFIDENCE: ", max_confidence, " CHAR = ", best_guess_char)
        reg += best_guess_char
        max_confidence = 0
        best_guess_char = ""

    print(reg.upper())
    return reg


""" start() method - accepts input image directory and runs ANPR pipeline stages, returning predicted match against input

    Toggleable Pre-processing Pipeline Stages:
    
    1a :- Noise Removal (Bilateral Filtering)
    1b :- Improving Contrast (Adaptive Histogram Equalisation)
    1c :- Noise Removal (Adaptive Gaussian Thresholding) (on) | Default: Otsu's Thresholding (off)
    1d :- Tilt Correction (Bilateral Transformation)"""


def start(image_list, image_dir, limit, s_1a, s_1b, s_1c, s_1d, plot_results):
    print("start() method pipeline stages enabled: ", s_1a, s_1b, s_1c, s_1d)
    begin_time = time.time()
    correct = 0
    incorrect_reg = []
    global stack
    stack = []

    limit = limit
    count = 0
    for file in image_list:
        print(file)
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

            # applying adaptative thresholding https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
            if s_1c:
                th_img = adaptive_threshold(ahe_img)
            else:
                th_val, th_img = cv2.threshold(ahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Character Segmentation
            th_img = cv2.bitwise_not(th_img)  # invert binary img OpenCV CCA expects black background, white foreground
            char_img, rect_border, characters = character_segmentation(th_img, s_1d)

            # Extract Characters from Original Input Image
            ext_char_templates = extract_characters(char_img, rect_border)

            # Template Matching
            reg = template_match(ext_char_templates)

            # Number Plate Assumption: (A1) The letter 'I'/'i' does not appear in NPs, only 1. (REF Gov Standards)
            # A2: The letter 'O' and number '0' are equivalent (REF Gov Standard)
            # Equivalent Character Assumptions (todo: use domain knowledge for this?)
            if "I" in file:
                file = file.replace("I", "1")
            if "O" in file:
                file = file.replace("O", "0")

            print("%s took %s seconds\n" % (file, time.time() - start_time))
            if reg.upper() == file[:7]:
                correct = correct + 1
            else:
                incorrect_reg.append([reg.upper(), file[:7]])
                # todo: store incorrect template images (all ext chars and see confidence values)
                # todo: create dir with UUIDs last 4 digits + date and show exploded diagram of all values

            """--- DISPLAY PROCESSED IMAGES --- 
                Contents are only displayed if -v command line arg is provided (verbose flag enabled)
                else, result metrics are pushed to display
            """
            plot_results = plot_results
            if plot_results:
                # IF -v (verbose flag enabled) ... show
                rows = 3
                cols = 3

                # OpenCV reads images BGR, matplotlib reads in RGB. Convert all images.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                greyscale_img = cv2.cvtColor(greyscale_img, cv2.COLOR_BGR2RGB)
                filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
                ahe_img = cv2.cvtColor(ahe_img, cv2.COLOR_BGR2RGB)
                th_img = cv2.cvtColor(th_img, cv2.COLOR_BGR2RGB)
                output = image.copy()

                angle = stack.pop()
                print("{:0.2f} deg of tilt".format(angle))

                # reference: drawing around component border
                # https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
                for r in rect_border:
                    # (x, y), (x + w, y + h)
                    cv2.rectangle(output, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 255, 0), 3)
                char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB)

                # Reference displaying multiple images in matplotlib subplots:
                # https://www.geeksforgeeks.org/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/
                # average image = 580x160 = 5.3 inches x 1.7 inches
                fig = plt.figure(figsize=(14, 5))

                i = 1
                for img, title in [[image, "Input Image " + file], [greyscale_img, "Greyscaled Input RGB Image"],
                                   [filtered_image, "Bilateral Filtered Image"],
                                   [ahe_img, "Adaptive Histogram Equalisation"],
                                   [th_img, "Adaptive Gaussian Thresholding"],
                                   [output, "Connected Components (Characters)"],
                                   [char_img, "Characters of " + file]]:
                    fig.add_subplot(rows, cols, i)
                    plt.imshow(img)
                    plt.title(title, {'fontname': 'Arial'})
                    plt.axis("off")
                    i = i + 1
                plt.text(850, 100, reg.upper(), fontsize="40", fontname='Arial', color="black")
                plt.subplots_adjust(hspace=0.4)
                plt.show()

            count = count + 1
            if count == limit:
                end_time = time.time() - begin_time
                avg_processing_time = end_time / limit
                """
                    --- Result Metrics Output: ---
                """
                avg_reading_accuracy = (correct / limit) * 100
                results_output = ("--- Result Metrics Output: ---\nAverage Reading Accuracy: {:0.2f}%\n"\
                                 "Total time taken for {} inputs: {:0.2f} seconds\n"\
                                 "Average time taken to process each input: {:0.2f} seconds\n"\
                                 "Incorrect Registrations {}/{} (Predicted, Actual): {}\n"\
                                 "Peak Signal to Noise-Ratio (PSNR) avg. : \n"
                                  "Mean Squared Error (MSE) avg. : (lower is better)\n".format(avg_reading_accuracy, limit,
                                                                                     end_time, avg_processing_time,
                                                                                     len(incorrect_reg),
                                                                                     limit, incorrect_reg))
                f = open("anpr_results.txt", "w")
                f.write(results_output)
                f.close()
                print(results_output)
                break


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

        image_list = None
        if os.path.exists(args.d):
            image_dir = args.d
            image_list = sorted(os.listdir(args.d))
            print("valid path: ", args.d)
        else:
            raise Exception("Error: Invalid directory path provided.")

        if args.l is None:
            limit = len(image_list)
        elif isinstance(args.l, int) and args.l <= len(image_list) and len(image_list) > 0:
            limit = args.l
            print("valid limit :- ", limit)
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

        print("PLOT RESULTS: ", plot_results)

        stages = parse_stage_args(args.s)

        if stages == [] and len(stages) == 0:
            # all stages are toggled off, call start with false enabled on all
            start(image_list, image_dir, limit, False, False, False, False, plot_results)
        elif len(stages) > 0:
            s_1a, s_1b, s_1c, s_1d = call_preprocessing_pipeline(stages)
            start(image_list, image_dir, limit, s_1a, s_1b, s_1c, s_1d, plot_results)

        print("\nSTAGES ", stages)
        print("arg len ", len(sys.argv))


cl_args_handler()
# start()

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
