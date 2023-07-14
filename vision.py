import numpy as np
import cv2
from hsv_filter import HsvFilter
from edge_filter import EdgeFilter

class Vision:
    #CONSTANTS
    TRACKBAR_WINDOW = "Trackbars"
     

    def __init__(self, inner_img_path, method=cv2.TM_CCOEFF_NORMED):
         self.inner_image = cv2.imread(inner_img_path, cv2.IMREAD_UNCHANGED)
         self.inner_width = self.inner_image.shape[1]
         self.inner_height = self.inner_image.shape[0]
         self.method = method

    def find(self, outer_image,  threshold=0.5, max_results=5):
        result = cv2.matchTemplate(outer_image, self.inner_image, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))

        #if there are no locations, reshapes empty array
        #so that it can be returned without an error
        if not locations:
            return np.array([], dtype=np.int32).reshape(0,4)
        
        rectangles = []
        for i in locations:
            rect = [int(i[0]), int(i[1]), self.inner_width, self.inner_height]
            rectangles.append(rect)

        rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.05)

        #limits number of rectangles for stability
        if len(rectangles) > max_results:
            print('Too many matches found. Truncating results')
            rectangles = rectangles[:max_results]
        return rectangles

    def get_click_points(self, rectangles):
        points = []
        if len(rectangles):
            for i in rectangles:
                center_x = i[0] + int(i[2]/2)
                center_y = i[1] + int(i[3]/2)
                points.append((center_x, center_y))
        #put this in main function
        #cv2.imshow('CS:BRO', self.outer_image)
        return points

    def draw_rectangels(self, outer_image, rectangles):
        if len(rectangles):
            for i in rectangles:
                cv2.rectangle(outer_image, (i[0], i[1]), (i[2] + i[0], i[3] + i[1]), (255,0,255),3)
        return outer_image

    def draw_crosshairs(self, outer_image, points):
        for (center_x, center_y) in points:
            cv2.drawMarker(outer_image, (center_x, center_y), (255,0,255,), markerType=cv2.MARKER_CROSS)
        return outer_image

    #create Gui window with controls for adjusting arugments in real time 
    def init_control_gui(self):
        #create and resize windows
        cv2.namedWindow(self.TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.TRACKBAR_WINDOW, 350, 700)
        
        #callback function required by trackbar
        def callback(position):
            pass
        #OpenCV scale for HSV is H: 0-179, S: 0-255, V: 0-255
        cv2.createTrackbar('hmin', self.TRACKBAR_WINDOW, 0, 179, callback)
        cv2.createTrackbar('smin', self.TRACKBAR_WINDOW, 0, 255, callback)
        cv2.createTrackbar('vmin', self.TRACKBAR_WINDOW, 0, 255, callback)
        cv2.createTrackbar('hmax', self.TRACKBAR_WINDOW, 0, 179, callback)
        cv2.createTrackbar('smax', self.TRACKBAR_WINDOW, 0, 255, callback)
        cv2.createTrackbar('vmax', self.TRACKBAR_WINDOW, 0, 255, callback)
        #Set default value for max HSV trackers
        cv2.setTrackbarPos('hmin', self.TRACKBAR_WINDOW, 179)
        cv2.setTrackbarPos('smin', self.TRACKBAR_WINDOW, 255)
        cv2.setTrackbarPos('vmin', self.TRACKBAR_WINDOW, 255)

        #trackbars for increasing and decreasing saturation and value
        cv2.createTrackbar('sadd', self.TRACKBAR_WINDOW, 0, 255, callback)
        cv2.createTrackbar('ssub', self.TRACKBAR_WINDOW, 0, 255, callback)
        cv2.createTrackbar('vadd', self.TRACKBAR_WINDOW, 0, 255, callback)
        cv2.createTrackbar('vsub', self.TRACKBAR_WINDOW, 0, 255, callback)

        #trackbars for edge creation
        cv2.createTrackbar('KernelSize', self.TRACKBAR_WINDOW, 1, 30,callback)
        cv2.createTrackbar('ErodeIter', self.TRACKBAR_WINDOW, 1, 5, callback)
        cv2.createTrackbar('DilateIter', self.TRACKBAR_WINDOW, 1, 5, callback)
        cv2.createTrackbar('canny1', self.TRACKBAR_WINDOW, 0, 200, callback)
        cv2.createTrackbar('canny2', self.TRACKBAR_WINDOW, 0, 500, callback)

        #set default value for canny trackbars
        cv2.setTrackbarPos('KernelSize', self.TRACKBAR_WINDOW, 5)
        cv2.setTrackbarPos('canny1', self.TRACKBAR_WINDOW, 100)
        cv2.setTrackbarPos('canny2', self.TRACKBAR_WINDOW, 200)

    #returns an HSV filter object based on the control GUI values
    def get_hsv_filter_from_controls(self):
        #get current position of all trackbars
        hsv_filter = HsvFilter()
        hsv_filter.hmin = cv2.getTrackbarPos('hmin', self.TRACKBAR_WINDOW)
        hsv_filter.smin = cv2.getTrackbarPos('smin', self.TRACKBAR_WINDOW)
        hsv_filter.vmin = cv2.getTrackbarPos('vmin', self.TRACKBAR_WINDOW)
        hsv_filter.hmax = cv2.getTrackbarPos('hmax', self.TRACKBAR_WINDOW)
        hsv_filter.smax = cv2.getTrackbarPos('smax', self.TRACKBAR_WINDOW)
        hsv_filter.vmax = cv2.getTrackbarPos('vmax', self.TRACKBAR_WINDOW)
        hsv_filter.sadd = cv2.getTrackbarPos('sadd', self.TRACKBAR_WINDOW)
        hsv_filter.ssub = cv2.getTrackbarPos('ssub', self.TRACKBAR_WINDOW)
        hsv_filter.vadd = cv2.getTrackbarPos('vadd', self.TRACKBAR_WINDOW)
        hsv_filter.vsub = cv2.getTrackbarPos('vsub', self.TRACKBAR_WINDOW)
        return hsv_filter
    
    def get_edge_filter_from_controls(self):
        #get current position of all trackers
        edge_filter = EdgeFilter()
        edge_filter.kernelSize = cv2.getTrackbarPos('KernelSize', self.TRACKBAR_WINDOW)
        edge_filter.erodeIter = cv2.getTrackbarPos('ErodeIter', self.TRACKBAR_WINDOW)
        edge_filter.dilateIter = cv2.getTrackbarPos('DilateIter', self.TRACKBAR_WINDOW)
        edge_filter.canny1 = cv2.getTrackbarPos('canny1',self.TRACKBAR_WINDOW)
        edge_filter.canny2 = cv2.getTrackbarPos('canny2', self.TRACKBAR_WINDOW)
        return edge_filter
    
    #Given an image and an HSV filter, apply the filter and return the resulting image.
    #If a filter is not supplied, the control GUI trackbars will be used.
    def apply_hsv_filter(self, original_image, hsv_filter=None):
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        if not hsv_filter:
            hsv_filter = self.get_hsv_filter_from_controls()

        #add / subtract saturation and value
        h,s,v =cv2.split(hsv)
        s = self.shift_channel(s, hsv_filter.sadd)
        s = self.shift_channel(s, -hsv_filter.ssub)
        v = self.shift_channel(v, hsv_filter.vadd)
        v = self.shift_channel(v, -hsv_filter.vsub)
        hsv = cv2.merge([h,s,v])
        
        lower = np.array([hsv_filter.hmin, hsv_filter.smin, hsv_filter.vmin])
        upper = np.array([hsv_filter.hmax, hsv_filter.smax, hsv_filter.vmax])

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(hsv, hsv, mask=mask)

        # Convert image back to BGR for imshow() to display properly
        img = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        return img

    #WE GOIN HARD NOW BOIIII
    #given an image and a Canny edge filter, apply the filter and return the resulting image.
    #if a filter is not supplied, the control gui trackbars will be used. 
    def apply_edge_filter(self, original_image, edge_filter=None):
        #if filter value is not defined, GUI filter values will be used
        if not edge_filter: edge_filter = self.get_edge_filter_from_controls()

        kernel = np.ones((edge_filter.kernelSize, edge_filter.kernelSize), np.uint8)
        eroded_image = cv2.erode(original_image, kernel, iterations=edge_filter.erodeIter)
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=edge_filter.dilateIter)

        #canny edge detection
        result = cv2.Canny(dilated_image, edge_filter.canny1, edge_filter.canny2)

        #convert single channel image back to BGR
        img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return img
    def shift_channel(self, c, amount):
        if amount > 0:
            lim = 255 - amount
            c[c >= lim] = 255
            c[c < lim] += amount
        elif amount < 0:
            amount = -amount
            lim = amount
            c[c <= lim] = 0
            c[c > lim] -= amount
        return c

    def match_keypoint(self, original_image, patch_size=32):
        min_match_count = 5

        orb = cv2.ORB_create(edgeThreshold=0, patchSize=patch_size)
        keypoints_needle, descriptors_needle = orb.detectAndCompute(self.inner_image, None)
        orb2 = cv2.ORB_create(edgeThreshold=0, patchSize=patch_size, nfeatures=2000)
        keypoints_haystack, descriptors_haystack = orb2.detectAndCompute(original_image, None)

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
        
        search_params = dict(checks=50)

        try:
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors_needle, descriptors_haystack, k=2)
        except cv2.error:
            return None, None, [], [], None
        
        good = []
        points = []
        matches_mask = None

        for pair in matches:
            if len(pair) == 2:
                if pair[0].distance < 0.7*pair[1].distance:
                    good.append(pair[0])

        if len(good) > min_match_count:
            print('match %03d, kp %03d' % (len(good), len(keypoints_needle)))
            for match in good:
                points.append(keypoints_haystack[match.trainIdx].pt)
        return keypoints_needle, keypoints_haystack, good, points, matches_mask

    def centroid(self, points_list):
        points_list = np.array(points_list, dtype=np.int32)
        length = points_list.shape[0]
        sum_x = np.sum(points_list[:,0])
        sum_y = np.sum(points_list[:,1])
        return [np.floor_divide(sum_x, length), np.floor_divide(sum_y, length)]