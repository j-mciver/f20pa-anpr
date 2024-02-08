import cv2
import os
import time
from matplotlib import pyplot as plt
import numpy as np

image_dir = "/Users/jmciver/Documents/Y4S1/F20PA/DISSERTATION-MATERIAL/UKLicencePlateDataset/whiteplate_augmented"
# image dir must not be hardcoded - make this an embedded project folder potentially on GitHub? + test yellow plates
image_list = sorted(os.listdir(image_dir))

# templates_dir = "/Users/jmciver/Documents/Y4S1/F20PA/DISSERTATION-MATERIAL/UKLicencePlateDataset/templates"
templates_dir = "/Users/jmciver/PycharmProjects/f20pa-anpr/templates"
templates_list = sorted(os.listdir(templates_dir))


def apply_bilateral_filter(img):
    return cv2.bilateralFilter(img, 7, 75, 75)


# reference: bilateral filter algorithm https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
IBF_THRESHOLD = 1.00


def iterative_bilateral_filter(img):
    fimg = apply_bilateral_filter(img)
    psnr = cv2.PSNR(img, fimg)
    print("PSNR: %s" % (psnr))
    return fimg


def tilt_correction(th_img):
    # 1 = cv2.RETR_EXTERNAL (exclude nested/internal contours)
    # 2 = cv2.CHAIN_APPROX_SIMPLE we want diagonal lines
    contours, _ = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    number_plate = cv2.minAreaRect(contours[7])
    box = cv2.boxPoints(number_plate)
    box = np.int0(box)
    output = cv2.cvtColor(th_img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(output, [box], 0, (0, 0, 255), 2)
    cv2.imshow("output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    centre = number_plate[0]
    (x, y) = number_plate[1]
    angle = number_plate[2]
    print("TILT CORRECTION HERE: ",number_plate)
    print(centre)
    print("x, y: ", (x, y))
    print("angle of rotation deg: ", angle)

    # (centre, (width, height), rot_angle) = number_plate
    return


def character_segmentation(th_img):
    # Reference: Connected-Component Analysis (https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/)
    output = cv2.connectedComponentsWithStats(th_img, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    char_img = np.zeros(th_img.shape, dtype="uint8")
    characters = []
    rect_border = []

    # sort components to appear based x-axis order (sorted on character, left to right)
    x_axis_sorted_components = list()
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        x_axis_sorted_components.append([x, i])

    list.sort(x_axis_sorted_components)
    # know that the left most component will be the number plate itself, take that out of the loop and fix each sorted_component image
    # sanity check: make sure that area of component is largest area out of every selected component
    # warpAffine...

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

        # area > 2500 == number plate rectangle ??
        # rotated rectangle: https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
        # if area > 1500:
            # output = cv2.cvtColor(th_img, cv2.COLOR_GRAY2RGB)
            # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # cv2.imshow("number plate", output)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # filter connected components by width, height and area of pixels
        if all((5 < w < 50, 40 < h < 65, 360 < area < 1500)):
            # output = cv2.cvtColor(th_img, cv2.COLOR_GRAY2RGB)
            # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)

            print("Keeping component {}".format(text))
            component_mask = (labels == j).astype("uint8") * 255
            characters.append(component_mask)
            rect_border.append([x, y, w, h])
            char_img = cv2.bitwise_or(char_img, component_mask)

            # cv2.imshow("Output", output)
            # cv2.waitKey(0)

    return char_img, rect_border, characters


def extract_characters(char_img, rect_border):
    extracted_char_templates = []

    # normalise extracted characters to be suitable for template matching
    for r in rect_border:
        # y:y + h, x:x + w (rows, columns)
        ext_char = char_img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
        ext_char = cv2.bitwise_not(ext_char)

        # resize to template width and height (30, 60)
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

            # todo: implement more intelligent logic against contolling confidence (will have to write assumption)
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


# Reference: OpenCV Converting RGB images to Greyscale: https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
def start():
    begin_time = time.time()
    correct = 0
    incorrect_reg = []

    limit = 1
    count = 0
    for file in image_list:
        print(file)
        start_time = time.time()
        if file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg"):
            print("Processing {0}".format(file))

            image_path = image_dir + "/" + file
            image = cv2.imread(image_path)
            greyscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # apply iterative bilateral filter
            filtered_image = iterative_bilateral_filter(greyscale_img)

            # adaptative histogram equalisation
            # AHE reference: https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/
            ahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(16, 16))
            ahe_img = ahe.apply(filtered_image)

            # apply otsu's method of automatic thresholding
            # reference: Applying Otsu's method of thresholding. https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
            th_val, th_img = cv2.threshold(ahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th_img = cv2.bitwise_not(th_img)  # invert binary img OpenCV CCA expects black background, white foreground

            # Tilt Correction
            # tilt_correction(th_img)

            # Character Segmentation
            char_img, rect_border, characters = character_segmentation(th_img)

            # Extract Characters from Original Input Image
            ext_char_templates = extract_characters(char_img, rect_border)

            # bilinear transformation - tilt detection and correction
            # resize() cv2 / pillow
            # correct tilt and then cmompare to template matching
            # correcting tilt can be done after, once we have the characters, means theres less to work with?

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

            """--- DISPLAY PROCESSED IMAGES --- 
                Contents are only displayed if -v command line arg is provided (verbose flag enabled)
                else, result metrics are pushed to display
            """
            plot_results = True
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

                # reference: drawing around component border
                # https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
                for r in rect_border:
                    # (x, y), (x + w, y + h)
                    cv2.rectangle(output, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 255, 0), 3)
                char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB)

                # Reference displaying multiple images in matplotlib subplots:
                # https://www.geeksforgeeks.org/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/
                # average image = 580x160 = 5.3 inches x 1.7 inches
                fig = plt.figure(figsize=(20, 6))

                i = 1
                for img, title in [[image, "Input Image " + file], [greyscale_img, "Greyscaled Input RGB Image"],
                                   [filtered_image, "Bilateral Filtered Image"],
                                   [ahe_img, "Adaptive Histogram Equalisation"],
                                   [th_img, "Automatic Thresholding (Otsu's Method)"],
                                   [output, "Connected Components (Characters)"],
                                   [char_img, "Characters of " + file]]:
                    fig.add_subplot(rows, cols, i)
                    plt.imshow(img)
                    plt.title(title)
                    plt.axis("off")
                    i = i + 1
                plt.text(850, 100, reg.upper(), fontsize="40", color="black")
                plt.subplots_adjust(hspace=0.5)
                plt.show()

            count = count + 1
            if count == limit:
                end_time = time.time() - begin_time
                """
                    --- Analytics / Result Metrics Output: ---
                """
                avg_reading_accuracy = (correct / limit) * 100
                print("--- Analytics / Result Metrics Output: ---\nAverage Reading Accuracy: {}%\n"
                      "Total time taken for {} inputs: {:0.2f} seconds\n"
                      "Incorrect Registrations {}/{} (Predicted, Actual): {}".format(avg_reading_accuracy, limit,
                                                                                     end_time, len(incorrect_reg),
                                                                                     limit, incorrect_reg))
                break


start()

# REFERENCES
# https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
# Bilateral Filter: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
# Peak-Signal to Noise Ratio: (OpenCV) https://shimat.github.io/opencvsharp_docs/html/23f56d6b-49ef-3365-5139-e75712c20fe4.htm
# Otsu's Method of Thresholding (OpenCV): https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
# https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/
# https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
# https://www.geeksforgeeks.org/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/
# https://learnopencv.com/cropping-an-image-using-opencv/#cropping-using-opencv
# resizing an image (opencv) tutorial reference: https://learnopencv.com/image-resizing-with-opencv/
# https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html (tilt correction based on affine transformation)
