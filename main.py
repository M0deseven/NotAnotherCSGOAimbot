import cv2, os
import numpy as np
import pyautogui
import pyfiglet
from winscreencapture import WindowCapture
import time
from vision import Vision
from hsv_filter import HsvFilter
from edge_filter import EdgeFilter
from tqdm import tqdm

main_banner = pyfiglet.figlet_format(text='CS:BRO')
print(main_banner)

with tqdm(total=100) as loading_bar:
    for i in range(100):
        time.sleep(0.073)
        loading_bar.update()
loading_bar.close()

wincap = WindowCapture('Counter-Strike: Global Offensive - Direct3D 9')

t_close_2 = Vision(r'python_stuff\computer_vision\ct_nofilter1.jpg')
#initialize trackbar window
t_close_2.init_control_gui()
#custom HSV filter for finding ct in dust 2 map
dust2_ct_filter = HsvFilter(21,172,0,26,237,255,187,63,208,0)


def main_loop():
    loop_time = time()
    while True:
        #gets up to date image of selected window
        screenshot = wincap.screen_cap()

        cv2.imshow('CS:BRO', screenshot)

        ##Applies hsv filters
        #processed_image = t_close_2.apply_hsv_filter(screenshot)

        ##Edge detection
        #edges = t_close_2.apply_edge_filter(processed_image)

        ##performs object detection
        #rectangles = t_close_2.find(processed_image, threshold=0.35)

        ##draw detection rectangles onto original image
        #output_image = t_close_2.draw_rectangels(screenshot, rectangles)


        ###experimental
        #keypoint searching
        #keypoint_image = edges 
        #kp1, kp2, matches, match_points, whatever= t_close_2.match_keypoint(keypoint_image)
        #match_image = cv2.drawMatches(
            #t_close_2.inner_image,
            #kp1,
            #keypoint_image,
            #kp2,
            #matches,
            #None)

        #if match_points:
            ##find the center point of all the matched features
            #center_point = t_close_2.centroid(match_points)
            ##account for width of needle image that appears on the left
            #center_point[0] += t_close_2.inner_width
            ##draw the found center point on the output image
            #match_image = t_close_2.draw_crosshairs(match_image, [center_point])

        #display processed image
        #calculate FPS by measuring delta between time() function calls
        print(f'FPS: {1/(time() - loop_time)}')
        loop_time = time()

        #handles keypresses and controls
        # 'q' will quit the program
        # 'f' will add a screenshot to the positive folder
        # 'd' will add a screenshot to the negative folder
        # waits 1ms after every loop to process keypress
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord('f'):
            cv2.imwrite(fr'python_stuff\computer_vision\positive\{loop_time}.jpg', screenshot)
        elif key == ord('d'):
            cv2.imwrite(fr'python_stuff\computer_vision\negative\{loop_time}.jpg', screenshot)



    print('done')
main_loop()