import cv2
import os
import time
from matplotlib import pyplot as plt
import numpy as np

image_dir = "/Users/jmciver/Documents/Y4S1/F20PA/DISSERTATION-MATERIAL/UKLicencePlateDataset/whiteplate_augmented"
# image dir must not be hardcoded - make this an embedded project folder potentially on GitHub? + test yellow plates
# warn: large file size on github repo?
image_list = sorted(os.listdir(image_dir))


def apply_bilateral_filter(img):
    return cv2.bilateralFilter(img, 7, 75, 75)


# reference: bilateral filter algorithm https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
IBF_THRESHOLD = 1.00


def iterative_bilateral_filter(img):
    fimg = apply_bilateral_filter(img)
    psnr = cv2.PSNR(img, fimg)
    print("PSNR: %s" % (psnr))
    return fimg


def character_segmentation(th_img):
    # Reference: Connected-Component Analysis (https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/)
    output = cv2.connectedComponentsWithStats(th_img, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    char_img = np.zeros(th_img.shape, dtype="uint8")
    characters = []
    rect_border = []

    for i in range(1, numLabels):
        text = "component {}/{}".format(i + 1, numLabels)

        # print a status message update for the current connected component
        print("[INFO] {}".format(text))

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        print("LABEL {}/{} stats: w = {}, h = {}, area = {}".format(i + 1, numLabels, w, h, area))

        # filter connected components by width, height and area of pixels
        if all((5 < w < 50, 40 < h < 65, 420 < area < 1500)):
            print("Keeping component {}".format(text))
            componentMask = (labels == i).astype("uint8") * 255
            characters.append(componentMask)
            rect_border.append([x, y, w, h])
            char_img = cv2.bitwise_or(char_img, componentMask)

    return char_img, rect_border, characters


def extract_characters(char_img, rect_border):
    extracted_char_templates = []

    # normalise extracted characters to be suitable for template matching
    for r in rect_border:
        # y:y + h, x:x + w (rows, columns)
        ext_char = char_img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]

        # remove background from image
        ext_char = cv2.bitwise_not(ext_char)
        ext_char = cv2.cvtColor(ext_char,
                                cv2.COLOR_GRAY2RGBA)  # convert to alpha to remove white pixels (set to transparent)

        for arr in ext_char:
            for row in arr:
                if np.array_equal(row[:3], [255, 255, 255]):
                    row[3] = 0  # set alpha value to transparent

        # resize to template width and height (30, 60)
        ext_char = cv2.resize(ext_char, (30, 60), cv2.INTER_LINEAR)

        extracted_char_templates.append(ext_char)

    return extracted_char_templates


# Reference: OpenCV Converting RGB images to Greyscale: https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
def start():
    limit = 1
    count = 0
    for file in image_list:
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

            # Character Segmentation
            char_img, rect_border, characters = character_segmentation(th_img)

            # Extract Characters from Original Input Image
            testIMG = extract_characters(char_img, rect_border)[0]

            # for char in characters:
            #     cv2.imshow("char", char)
            #     cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # bilinear transformation - tilt detection and correction
            # resize() cv2 / pillow
            # correct tilt and then cmompare to template matching
            # correcting tilt can be done after, once we have the characters, means theres less to work with?

            print("%s took %s seconds\n" % (file, time.time() - start_time))

        """--- DISPLAY PROCESSED IMAGES --- 
            Contents are only displayed if -v command line arg is provided (verbose flag enabled)
            else, result metrics are pushed to display
        """
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
        # average image = 580x160 = 5.3 inches x 1.7
        fig = plt.figure(figsize=(20, 6))

        i = 1
        for img, title in [[image, "Input Image " + file], [greyscale_img, "Greyscaled Input RGB Image"],
                           [filtered_image, "Bilateral Filtered Image"],
                           [ahe_img, "Adaptive Histogram Equalisation"],
                           [th_img, "Automatic Thresholding (Otsu's Method)"],
                           [output, "Connected Components (Characters)"],
                           [char_img, "Characters of " + file],
                           [testIMG, "extracted template"]]:
            fig.add_subplot(rows, cols, i)
            plt.title(title)
            plt.axis("off")
            i = i + 1

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        count = count + 1
        if count == limit:
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
