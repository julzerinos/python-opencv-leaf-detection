import cv2 as cv  # opencv
import copy  # for deepcopy on images

import numpy as np  # numpy
from random import randint  # for random values

import threading  # for deamon processing

from pathlib import Path  # for directory information
import os  # for directory information

from constants import constants  # constants


class PlantDetector:
    """Dynamically apply detection algorithms to source images

    All images are sourced from and follow naming standard from
    the KOMATSUNA dataset
    http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/

    METHODS

    __init__(self, src='multi_plant', labels='multi_label') [void]
        prepares the images and labels for display
        initializes windows and trackbars
        runs background subtraction on plant group images

    on_low_H_thresh_trackbar(self, val)
    on_high_H_thresh_trackbar(self, val)
    on_low_S_thresh_trackbar(self, val)
    on_high_S_thresh_trackbar(self, val)
    on_low_V_thresh_trackbar(self, val)
    on_high_V_thresh_trackbar(self, val)
        HSV trackbar triggers

    prepare_plant_collection(self, src, labelsrc)
        returns [plants, plant_groups, labels]
        constructor helper function for loading plant images

    parse(self, auto_inc=False, mode=0) [void]
        main function
        dynamically applies
            HSV inRange filters
            watershed algorithm
        to the currently displayed image
        based on selected HSV trackbar values

        six modes are displayable:

        mode: window1               + window2
        0   : original (fallback)   + original
        1   : HSV filter range      + original
        2   : bare watershed masks  + labels
        3   : watershed masks w/ bg + original
        4   : sequential bg sub     + original
        5   : seq bg sub w/ watersh + original

        additionally, the user is allowed control
        key | function
        m   | next image
        n   | prev image
        s   | save selected image in the selected mode
        z   | save all images in selected mode
        esc | exit the program
        d   | dynamically calculate dice
        f   | show dice data based on saved images
        1-5 | select the respective mode

        parse is also used for saving all images
        parse is run for all images in the given mode
        either in parrallel or in place

    save_one(self, mode, image, filename) [void]
        saves the image in the appropriate mode folder with filename

    HSV_filtering_and_watershed(self, input_im) [mask, input_im, im_threshold]
        image is filtered through HSV inRange according to trackbar values
        image is prepared (threshold) for watershed algorithm
        watershed algorithm is applied
        markers are applied to image

    dicify_wrapper(self, image_id) [void]
        runs dice summary in background

    dicify_summary(self, image_id) [void]
        prints summary of dice values for image, plant, dateset
        note: based on saved images

    dicify_one(self, image_id) [dice]
        returns the dice value for the given image_id
        based on saved segmentation and label images

    dicify_one_dynamic(self, mask, image_id) [dice]
        returns dice value for the given image_id
        based on given mask (current) and saved label image

    dicify_plant(self, plant_id) [mean, min, max]
        returns mean, min and max dice values for images in plant group

    dicify_all(self) [mean, min, max]
        returns mean, min and max dice values for images in dataset
        and for each plant
    """

    def __init__(self, src='multi_plant', labels='multi_label'):
        self.c = constants()
        self.window1 = self.c.window.window1
        self.window2 = self.c.window.window2

        cv.namedWindow(self.window1)
        cv.namedWindow(self.window2)

        cv.moveWindow(self.window2, 550, 90)

        cv.createTrackbar(
            self.c.HSV.low_H_name, self.window1, self.c.HSV.low_H,
            self.c.HSV.max_value_H, self.on_low_H_thresh_trackbar)
        cv.createTrackbar(
            self.c.HSV.high_H_name, self.window1, self.c.HSV.high_H,
            self.c.HSV.max_value_H, self.on_high_H_thresh_trackbar)
        cv.createTrackbar(
            self.c.HSV.low_S_name, self.window1, self.c.HSV.low_S,
            self.c.HSV.max_value, self.on_low_S_thresh_trackbar)
        cv.createTrackbar(
            self.c.HSV.high_S_name, self.window1, self.c.HSV.high_S,
            self.c.HSV.max_value, self.on_high_S_thresh_trackbar)
        cv.createTrackbar(
            self.c.HSV.low_V_name, self.window1, self.c.HSV.low_V,
            self.c.HSV.max_value, self.on_low_V_thresh_trackbar)
        cv.createTrackbar(
            self.c.HSV.high_V_name, self.window1, self.c.HSV.high_V,
            self.c.HSV.max_value, self.on_high_V_thresh_trackbar)

        self.plants, self.plant_groups, self.labels = self.prepare_plant_collection(src, labels)

        # source https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html

        for key in self.plant_groups:
            if self.c.bgsub.mod == 'MOG2':
                backSub = cv.createBackgroundSubtractorMOG2(history=60, detectShadows=True)
            elif self.c.bgsub.mod == 'KNN':
                backSub = cv.createBackgroundSubtractorKNN()
            fgMask = None
            for i, image in enumerate(self.plant_groups[key]):
                fgMask = backSub.apply(image)
                self.plant_groups[key][i] = fgMask

    def on_low_H_thresh_trackbar(self, val):
        self.c.HSV.low_H = val
        self.c.HSV.low_H = min(self.c.HSV.high_H-1, self.c.HSV.low_H)
        cv.setTrackbarPos(
            self.c.HSV.low_H_name, self.window1, self.c.HSV.low_H)

    def on_high_H_thresh_trackbar(self, val):
        self.c.HSV.high_H = val
        self.c.HSV.high_H = max(self.c.HSV.high_H, self.c.HSV.low_H+1)
        cv.setTrackbarPos(
            self.c.HSV.high_H_name, self.window1, self.c.HSV.high_H)

    def on_low_S_thresh_trackbar(self, val):
        self.c.HSV.low_S = val
        self.c.HSV.low_S = min(self.c.HSV.high_S-1, self.c.HSV.low_S)
        cv.setTrackbarPos(
            self.c.HSV.low_S_name, self.window1, self.c.HSV.low_S)

    def on_high_S_thresh_trackbar(self, val):
        self.c.HSV.high_S = val
        self.c.HSV.high_S = max(self.c.HSV.high_S, self.c.HSV.low_S+1)
        cv.setTrackbarPos(
            self.c.HSV.high_S_name, self.window1, self.c.HSV.high_S)

    def on_low_V_thresh_trackbar(self, val):
        self.c.HSV.low_V = val
        self.c.HSV.low_V = min(self.c.HSV.high_V-1, self.c.HSV.low_V)
        cv.setTrackbarPos(
            self.c.HSV.low_V_name, self.window1, self.c.HSV.low_V)

    def on_high_V_thresh_trackbar(self, val):
        self.c.HSV.high_V = val
        self.c.HSV.high_V = max(self.c.HSV.high_V, self.c.HSV.low_V+1)
        cv.setTrackbarPos(
            self.c.HSV.high_V_name, self.window1, self.c.HSV.high_V)

    def prepare_plant_collection(self, src, labelsrc):
        plants = []
        plant_groups = dict()
        files = os.listdir(src)
        files.sort()
        for fl in files:
            input_im = cv.imread(src + '/' + fl, cv.IMREAD_COLOR)
            if (input_im is None):
                exit()

            plants.append({
                'p': input_im,
                'n': fl
            })
            group_id = f'{fl.split("_")[1]}{fl.split("_")[2]}'
            if group_id not in plant_groups:
                plant_groups[group_id] = []
            plant_groups[group_id].append(input_im)

        labels = []
        files = os.listdir(labelsrc)
        files.sort()
        for fl in files:
            input_im = cv.imread(labelsrc + '/' + fl)
            if (input_im is None):
                exit()
            labels.append(input_im)

        return plants, plant_groups, labels

    def parse(self, auto_inc=False, mode=0):
        key = 0
        i = 0
        l_tog = False

        while key != self.c.cntr.exit_k:

            if auto_inc and i == len(self.plants):
                break

            image = copy.deepcopy(self.plants[i]['p'])
            group_id = f'{self.plants[i]["n"].split("_")[1]}{self.plants[i]["n"].split("_")[2]}'

            mask, markers, im_threshold = self.HSV_filtering_and_watershed(image)

            _, bgfgSegMarkers, _ = self.HSV_filtering_and_watershed(
                        cv.cvtColor(self.plant_groups[group_id][i % 60], cv.COLOR_GRAY2BGR)
                        )

            if mode == 5:
                alt = bgfgSegMarkers
                text = f'Watershed new areas w/ fg/bg segm. {self.plants[i]["n"]}'
                tcol = (255, 255, 255)
            elif mode == 4:
                alt = copy.deepcopy(self.plant_groups[group_id][i % 60])
                text = f'FG/BG segmentation {self.plants[i]["n"]}'
                tcol = (255, 255, 255)
            elif mode == 3:
                alt = markers
                text = f'Watershed algorithm areas w/ bg {self.plants[i]["n"]}'
                tcol = (0, 0, 0)
            elif mode == 2:
                alt = mask
                text = f'Watershed algorithm areas bare {self.plants[i]["n"]}'
                tcol = (255, 255, 255)
            elif mode == 1:
                alt = im_threshold
                text = f'HSV inRange threshold {self.plants[i]["n"]}'
                tcol = (255, 255, 255)
            else:
                alt = copy.deepcopy(self.plants[i]['p'])
                text = f'Original {self.plants[i]["n"]}'
                tcol = (0, 0, 0)

            if self.c.asth.text:
                cv.putText(alt, text, (0, 20), self.c.asth.font, .5, tcol, 1)

            cv.imshow(self.window1, alt)
            if l_tog:
                cv.imshow(self.window2, self.labels[i])
            else:
                cv.imshow(self.window2, self.plants[i]['p'])

            key = cv.waitKey(10)

            if key == self.c.cntr.prev_k and i > 0:
                i -= 1
            if key == self.c.cntr.next_k and i < len(self.plants) - 1:
                i += 1
            if key == self.c.cntr.save or auto_inc:
                self.save_one(mode, alt, self.plants[i]["n"])
            if key == self.c.cntr.save_all:
                self.parse(True, mode)
            if key == self.c.cntr.dice:
                print(self.dicify_one_dynamic(mask, self.plants[i]['n']))
            if key == self.c.cntr.dice_more:
                self.dicify_wrapper(self.plants[i]['n'])

            if key == self.c.cntr.m1_k:
                mode = 1
                l_tog = False
            elif key == self.c.cntr.m2_k:
                mode = 2
                l_tog = True
            elif key == self.c.cntr.m3_k:
                mode = 3
                l_tog = False
            elif key == self.c.cntr.m4_k:
                mode = 4
                l_tog = False
            elif key == self.c.cntr.m5_k:
                mode = 5
                l_tog = False

            if auto_inc:
                i += 1

    def save_one(self, mode, image, filename):
        Path(f'formatted/{self.c.cntr.modes[mode]}').mkdir(parents=True, exist_ok=True)
        cv.imwrite(f'formatted/{self.c.cntr.modes[mode]}/{filename}', image)

    def HSV_filtering_and_watershed(self, input_im):

        im_threshold = cv.inRange(
            cv.cvtColor(input_im, cv.COLOR_BGR2HSV),
            (self.c.HSV.low_H, self.c.HSV.low_S, self.c.HSV.low_V),
            (self.c.HSV.high_H, self.c.HSV.high_S, self.c.HSV.high_V)
            )

        # source https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html

        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(im_threshold, cv.MORPH_OPEN, kernel, iterations=5)

        sure_bg = cv.dilate(opening, kernel, iterations=7)

        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        _, sure_fg = cv.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        _, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv.watershed(input_im, markers)

        input_im[markers == -1] = [255, 0, 0]
        for i in range(2, markers.max() + 1):
            input_im[markers == i] = [
                randint(0, 255), randint(0, 255), randint(0, 255)
                ] if self.c.xtra.disco else [
                    (40 + i * 40) % 255, (i * 40) % 255, (50 + i * 40) % 255
                ]

        mask = copy.deepcopy(input_im)
        mask[markers < 2] = [0, 0, 0]

        return mask, input_im, im_threshold

    def dicify_wrapper(self, image_id):
        thread = threading.Thread(target=self.dicify_summary, args=(image_id,), daemon=True)
        thread.start()

    def dicify_summary(self, image_id):
        print(self.dicify_all())

    def dicify_one(self, image_id):

        # Source: https://github.com/Kornelos/CV_MINI_1/blob/master/process_plants.py

        img = cv.imread(f'multi_label/label_{image_id.split("_", 1)[1]}')
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, gt = cv.threshold(img, 1, 255, cv.THRESH_BINARY)

        img = cv.imread(f'formatted/ws_mask/{image_id}')
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, rt = cv.threshold(img, 1, 255, cv.THRESH_BINARY)

        k = 255

        dice = np.sum(rt[gt == k]) * 2.0 / (np.sum(rt[rt == k]) + np.sum(gt[gt == k]))

        return dice

    def dicify_one_dynamic(self, mask, image_id):
        img = cv.imread(f'multi_label/label_{image_id.split("_", 1)[1]}')
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, gt = cv.threshold(img, 1, 255, cv.THRESH_BINARY)

        img = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        _, rt = cv.threshold(img, 1, 255, cv.THRESH_BINARY)

        k = 255

        dice = np.sum(rt[gt == k]) * 2.0 / (np.sum(rt[rt == k]) + np.sum(gt[gt == k]))

        return dice

    def dicify_plant(self, plant_id):
        vals = []
        for im_data in [
            t for t in self.plants
            if t['n'].split('_')[2] == plant_id
                ]:
            vals.append(self.dicify_one(im_data['n']))
        return [np.mean(vals), min(vals), max(vals)]

    def dicify_all(self):
        means = []
        mins = []
        maxs = []
        summ = "id | mean  | min   | max"
        for i in range(0, 5):
            plant = self.dicify_plant(f'0{str(i)}')
            means.append(plant[0])
            mins.append(plant[1])
            maxs.append(plant[2])
            summ += f'\n0{str(i)} | {round(plant[0], 3)} | {round(plant[1], 3)} | {round(plant[2], 3)}'
        summ += f'\nsm | {round(np.mean(means), 3)} | {round(min(mins), 3)} | {round(max(maxs), 3)}'
        return summ


# Main
plDt = PlantDetector()
plDt.parse()
