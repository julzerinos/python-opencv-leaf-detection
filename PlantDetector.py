import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plot
import copy


class constants:
    class HSV:
        max_value = 255
        max_value_H = 360//2

        low_H = 30
        low_S = 30
        low_V = 30
        high_H = 75
        high_S = 255
        high_V = 255

        low_H_name = 'Low H'
        low_S_name = 'Low S'
        low_V_name = 'Low V'
        high_H_name = 'High H'
        high_S_name = 'High S'
        high_V_name = 'High V'

    class window:
        window1 = 'Altered'
        window2 = 'Original'

    class asth:
        font = cv.FONT_HERSHEY_SIMPLEX

    class cntr:
        next_k = ord('m')
        prev_k = ord('n')
        exit_k = 27

        m1_k = ord('1')
        m2_k = ord('2')
        m3_k = ord('3')
        m4_k = ord('4')
        m5_k = ord('5')

    class flow:
        HSV_filtering_and_watershed = True
        bg_fg_segm = False


class PlantDetector:
    
    def on_low_H_thresh_trackbar(self, val):
        self.c.HSV.low_H = val
        self.c.HSV.low_H = min(self.c.HSV.high_H-1, self.c.HSV.low_H)
        cv.setTrackbarPos(self.c.HSV.low_H_name, self.window1, self.c.HSV.low_H)
    def on_high_H_thresh_trackbar(self, val):
        self.c.HSV.high_H = val
        self.c.HSV.high_H = max(self.c.HSV.high_H, self.c.HSV.low_H+1)
        cv.setTrackbarPos(self.c.HSV.high_H_name, self.window1, self.c.HSV.high_H)
    def on_low_S_thresh_trackbar(self, val):
        self.c.HSV.low_S = val
        self.c.HSV.low_S = min(self.c.HSV.high_S-1, self.c.HSV.low_S)
        cv.setTrackbarPos(self.c.HSV.low_S_name, self.window1, self.c.HSV.low_S)
    def on_high_S_thresh_trackbar(self, val):
        self.c.HSV.high_S = val
        self.c.HSV.high_S = max(self.c.HSV.high_S, self.c.HSV.low_S+1)
        cv.setTrackbarPos(self.c.HSV.high_S_name, self.window1, self.c.HSV.high_S)
    def on_low_V_thresh_trackbar(self, val):
        self.c.HSV.low_V = val
        self.c.HSV.low_V = min(self.c.HSV.high_V-1, self.c.HSV.low_V)
        cv.setTrackbarPos(self.c.HSV.low_V_name, self.window1, self.c.HSV.low_V)
    def on_high_V_thresh_trackbar(self, val):
        self.c.HSV.high_V = val
        self.c.HSV.high_V = max(self.c.HSV.high_V, self.c.HSV.low_V+1)
        cv.setTrackbarPos(self.c.HSV.high_V_name, self.window1, self.c.HSV.high_V)

    def prepare_plant_collection(self, src = 'multi_plant'):
        plants = []
        plant_groups = dict()
        files = os.listdir( src )
        files.sort()
        for fl in files:
            input_im = cv.imread(src + '/' + fl, cv.IMREAD_COLOR)
            if ( input_im is None ):
                exit()

            plants.append({
                'p': input_im,
                'n': fl
            })
            if fl.split('_')[2] not in plant_groups:
                plant_groups[fl.split('_')[2]] = []
            plant_groups[fl.split('_')[2]].append(input_im)

        return plants, plant_groups

    def __init__(self, src = 'multi_plant'):
        self.c = constants()
        self.window1 = self.c.window.window1
        self.window2 = self.c.window.window2

        cv.namedWindow(self.window1)
        cv.namedWindow(self.window2)
        
        cv.moveWindow(self.window2, 800, 0)

        cv.createTrackbar(self.c.HSV.low_H_name, self.window1 , self.c.HSV.low_H, self.c.HSV.max_value_H, self.on_low_H_thresh_trackbar)
        cv.createTrackbar(self.c.HSV.high_H_name, self.window1 , self.c.HSV.high_H,self.c.HSV.max_value_H, self.on_high_H_thresh_trackbar)
        cv.createTrackbar(self.c.HSV.low_S_name, self.window1 , self.c.HSV.low_S, self.c.HSV.max_value, self.on_low_S_thresh_trackbar)
        cv.createTrackbar(self.c.HSV.high_S_name, self.window1 , self.c.HSV.high_S,self.c.HSV.max_value, self.on_high_S_thresh_trackbar)
        cv.createTrackbar(self.c.HSV.low_V_name, self.window1 , self.c.HSV.low_V, self.c.HSV.max_value, self.on_low_V_thresh_trackbar)
        cv.createTrackbar(self.c.HSV.high_V_name, self.window1 , self.c.HSV.high_V,self.c.HSV.max_value, self.on_high_V_thresh_trackbar)

        self.plants, self.plant_groups = self.prepare_plant_collection(src)

        if self.c.flow.bg_fg_segm is False:
            return

        if False:
            backSub = cv.createBackgroundSubtractorMOG2()
        else:
            backSub = cv.createBackgroundSubtractorKNN()

        for key in self.plant_groups:
            if True:
                backSub = cv.createBackgroundSubtractorMOG2(history = 60, detectShadows=True)
            else:
                backSub = cv.createBackgroundSubtractorKNN()
            fgMask = None
            for i, image in enumerate(self.plant_groups[key]):
                fgMask = backSub.apply(image)
                self.plant_groups[key][i] = fgMask
            # self.plant_groups[key] = backSub.getBackgroundImage()

    def parse(self):
        key = 0
        mode = 0
        i = 0

        while key != self.c.cntr.exit_k:
            
            image = copy.deepcopy(self.plants[i]['p'])

            if self.c.flow.HSV_filtering_and_watershed:
                if not self.c.flow.bg_fg_segm:
                    wt_image, markers, im_threshold = self.HSV_filtering_and_watershed(image)
                else:
                    wt_image, markers, im_threshold = self.HSV_filtering_and_watershed(
                        cv.cvtColor(self.plant_groups[self.plants[i]['n'].split('_')[2]][i % 60], cv.COLOR_GRAY2BGR)
                        )

            if mode == 5 and self.c.flow.bg_fg_segm:
                alt = image
                mask = self.plant_groups[self.plants[i]['n'].split('_')[2]]
                alt[mask == 255] = [0, 0, 0]
            elif mode == 4 and self.c.flow.bg_fg_segm:
                alt = self.plant_groups[self.plants[i]['n'].split('_')[2]][i % 60]
                text = f'Fore/Background segmentation {self.plants[i]["n"]}'
                tcol = (255,255,255)
            elif mode == 3 and self.c.flow.HSV_filtering_and_watershed:
                alt = markers
                text = f'Watershed algorithm areas {self.plants[i]["n"]}'
                tcol = (0, 0, 0)

            elif mode == 2 and self.c.flow.HSV_filtering_and_watershed:
                alt = wt_image
                text = f'Watershed algorithm borders {self.plants[i]["n"]}'
                tcol = (0, 0, 0)
            elif mode == 1 and self.c.flow.HSV_filtering_and_watershed:
                alt = im_threshold
                text = f'HSV inRange threshold {self.plants[i]["n"]}'
                tcol = (255,255,255)
            else:
                alt = self.plants[i]['p']
                text = f'Original {self.plants[i]["n"]}'
                tcol = (0, 0, 0)

            cv.putText(alt, text, (0,20), self.c.asth.font, .5, tcol, 1)
            cv.putText(self.plants[i]['p'], 'Original', (0,20), self.c.asth.font, .5, (0,0,0), 1)

            cv.imshow(self.window1, alt)
            cv.imshow(self.window2, self.plants[i]['p'])

            key = cv.waitKey(10)

            if key == self.c.cntr.prev_k and i > 0:
                i -= 1
            if key == self.c.cntr.next_k and i < len(self.plants) - 1:
                i += 1

            if key == self.c.cntr.m1_k:
                mode = 1
            elif key == self.c.cntr.m2_k:
                mode = 2
            elif key == self.c.cntr.m3_k:
                mode = 3
            elif key == self.c.cntr.m4_k:
                mode = 4
            elif key == self.c.cntr.m5_k:
                mode = 5

    def HSV_filtering_and_watershed(self, input_im):
        
        im_threshold = cv.inRange(cv.cvtColor(input_im, cv.COLOR_BGR2HSV), (self.c.HSV.low_H, self.c.HSV.low_S, self.c.HSV.low_V), (self.c.HSV.high_H, self.c.HSV.high_S, self.c.HSV.high_V))

        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(im_threshold,cv.MORPH_OPEN,kernel, iterations = 2)

        sure_bg = cv.dilate(opening,kernel,iterations=7)

        dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
        _, sure_fg = cv.threshold(dist_transform,0.3*dist_transform.max(),255,0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg,sure_fg)

        # Marker labelling
        _, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv.watershed(input_im,markers)
        input_im[markers == -1] = [255,0,0]

        fig = plot.figure()
        plot.imshow(markers)
        fig.canvas.draw()

        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        markers = cv.cvtColor(img,cv.COLOR_RGB2BGR)

        plot.close(fig)
        
        return input_im, markers, im_threshold


# Main

plDt = PlantDetector()
plDt.parse()

