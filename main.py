import cv2
import os
import time


image_dir = "/Users/jmciver/Documents/Y4S1/F20PA/DISSERTATION-MATERIAL/UKLicencePlateDataset/whiteplate_augmented"
image_list = sorted(os.listdir(image_dir))


limit = 1
count = 0
# Reference: OpenCV Converting RGB images to Greyscale: https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
for file in image_list:
    start_time = time.time()
    if file.endswith(".png"):
        print("Converting {0} to greyscale".format(file))

        image_path = image_dir + "/" + file
        image = cv2.imread(image_path)
        greyscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("converted input RGB image", greyscale_img) #todo: make optional to show (on verbose flag enabled)
        # todo: matplotlib for all of them showing inputs and outputs etc

        # reference: Applying Otsu's method of thresholding. https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
        th_val, th_img = cv2.threshold(greyscale_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        title = "Otsu's Method: " + file + " threshold value: " + str(th_val)
        cv2.imshow("Input RGB image", image)
        cv2.imshow(title, th_img)
        # cv2.imshow("otsu's method of automatic thresholding", th_img)

        print("%s took %s seconds" % (file, time.time() - start_time))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        count = count + 1

    if count == limit:
        break

# REFERENCES
# https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
# Otsu's Method of Thresholding (OpenCV): https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html


