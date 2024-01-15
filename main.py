import cv2
import os

image_dir = "/Users/jmciver/Documents/Y4S1/F20PA/DISSERTATION-MATERIAL/UKLicencePlateDataset/whiteplate_augmented"
image_list = sorted(os.listdir(image_dir))


limit = 1
count = 0
# Reference: OpenCV Converting RGB images to Greyscale: https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
for file in image_list:
    if file.endswith(".png"):
        print("Converting {0} to greyscale".format(file))

        image_path = image_dir + "/" + file
        image = cv2.imread(image_path)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("converted input RGB image", grey) #todo: make optional to show (on verbose flag enabled)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        count = count + 1

    if count == limit:
        break



# REFERENCES
# https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/ (lines 17-21)