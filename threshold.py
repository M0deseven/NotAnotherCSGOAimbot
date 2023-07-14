import os, sys, cv2
import numpy as np

outer_img = cv2.imread(r'python_stuff\computer_vision\needle_and_haystack\csgoct.png')
inner_img = cv2.imread(r'python_stuff\computer_vision\needle_and_haystack\ct.png')

result = cv2.matchTemplate(outer_img, inner_img, cv2.TM_CCOEFF_NORMED)
print(result)

threshold = 0.47
locations = np.where(result >= threshold)


locations = list(zip(*locations[::-1]))
print(locations)

if locations:
    print('found match(es)')
    inner_width = inner_img.shape[1]
    inner_height = inner_img.shape[0]
    for i in locations:
        top_left = i
        bottom_right = (top_left[0] + inner_width, top_left[1] + inner_height)
        cv2.rectangle(outer_img, top_left, bottom_right, (255,0,255),3)
    cv2.imshow('matches', outer_img)
    cv2.waitKey()