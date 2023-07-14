import cv2, os
import numpy as np


OuterImage = cv2.imread(r'python_stuff\computer_vision\targeting\csgoct.png')
InnerImage = cv2.imread(r'python_stuff\computer_vision\targeting\ct.png')

result = cv2.matchTemplate(OuterImage, InnerImage, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

threshold = 0.8
if max_val >= threshold: 
    #get dimensions of inner images
    target_width = InnerImage.shape[1]
    target_height = InnerImage.shape[0]
    #apply image dimensions to coordinates to create rectangle
    top_left = max_loc
    bottom_right = (top_left[0] + target_width, top_left[1] + target_height)
    #create rectangle using coordinates and dimensions from previous step
    cv2.rectangle(OuterImage, top_left, bottom_right, (255,0,255), 3)
    cv2.imshow('target', OuterImage)
    cv2.waitKey()
    print('write image?')
    answer = input('//>')
    if answer.lower() == 'y':
        print('writing image')
        cv2.imwrite('result.jpg', OuterImage)
    else: print('skipping write')
else: print('no match found')