import csv

# input_registration, predicted, confidence, is_correct, np_extracted, psnr1, psnr2, psnr3, degree_of_tilt, pre_ahe_contrast, post_ahe_contrast,
""""
<input_registration_text> AA10QYN </input_registration_text>
<predicted_text> AA10QYN </predicted_text>
<max_confidence>
    <1st_char>90</1st_char>
    <2nd_char>80</2nd_char>
            ...
</max_confidence>

<confidence_distribution>
    <a>90,90,82,84..</a>
    <b>10,10,10,90..</b>
    <c>
    + 0-9 digits
     .. 7 items in every tag (for 7 characters in plate) show distribution for each predicted match
</confidence_distribution>

<is_correct>true</is_correct>
psnr
degree_of_tilt
contrast
illumination


"""
def write_results():
    return

# References:
# https://docs.python.org/3/library/csv.html