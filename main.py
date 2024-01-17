import cv2
import os
import time

image_dir = "/Users/jmciver/Documents/Y4S1/F20PA/DISSERTATION-MATERIAL/UKLicencePlateDataset/whiteplate_augmented"
image_list = sorted(os.listdir(image_dir))

IBF_THRESHOLD = 1.00


def apply_bilateral_filter(img):
    return cv2.bilateralFilter(img, 7, 75, 75)


# reference: bilateral filter algorithm https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
# todo: perhaps just apply bilateral filter once? - not seeing an improvement in noise reduction. could be due to nature of computer generated dataeset. compare to Kaur 2016.
def iterative_bilateral_filter(img):
    fimg = apply_bilateral_filter(img)
    psnr = cv2.PSNR(img, fimg)
    prev_psnr = psnr
    print("PSNR: %s, PREV_PSNR %s" % (psnr, prev_psnr))

    diff = 2**31
    # assess PSNR between iterations until convergence point is met. then stop execution.
    # if PSNR difference is below Iterative Bilateral Filtering Threshold (IBF_THRESHOLD), stop executing
    while diff > IBF_THRESHOLD:
        updated = apply_bilateral_filter(fimg)
        psnr = cv2.PSNR(fimg, updated)
        diff = psnr - prev_psnr
        print("PREV_PSNR: %s, PSNR: %s | diff = %s" % (prev_psnr, psnr, diff))
        if psnr > prev_psnr:  # previous filtered result was higher quality, do not filter more
            return fimg
        prev_psnr = psnr
        fimg = updated

    return updated


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
            # cv2.imshow("converted input RGB image", greyscale_img) #todo: make optional to show (on verbose flag enabled)
            # todo: matplotlib for all of them showing inputs and outputs etc

            # apply iterative bilateral filter
            filtered_image = iterative_bilateral_filter(greyscale_img)
            cv2.imshow("Iterative iterative_bilateral_filter Filter", filtered_image)

            # apply otsu's method of automatic thresholding
            # reference: Applying Otsu's method of thresholding. https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
            th_val, th_img = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            title = "Otsu's Method: " + file + " threshold value: " + str(th_val)
            cv2.imshow("Input RGB image", image)
            cv2.imshow(title, th_img)

            # adaptative histogram equalisation

            # bilinear transformation


            print("%s took %s seconds" % (file, time.time() - start_time))


            # todo: retire cv2 showwindow and apply into windowed / structured matplotlib instead.
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            count = count + 1

        if count == limit:
            break

start()

# REFERENCES
# https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
# Bilateral Filter: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
# Peak-Signal to Noise Ratio: (OpenCV) https://shimat.github.io/opencvsharp_docs/html/23f56d6b-49ef-3365-5139-e75712c20fe4.htm
# Otsu's Method of Thresholding (OpenCV): https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
