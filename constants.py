class constants:
    """Class of constants for each component of detector
    """

    class bgsub:
        """Background subtraction/segmentation

        mod [str] the segmentation model (MOG2, KNN, GMG)
        """
        mod = 'MOG2'

    class HSV:
        """HSV inRange filtering

        maximum values and initial values
        """
        max_value = 255
        max_value_H = 360//2

        low_H = 40
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
        """Window control

        names of windows
        """
        window1 = 'Altered'
        window2 = 'Original'

    class asth:
        """Aesthetics

        font [enum int] font used for description
        text [bool] should text be imprinted on image?
        """
        font = 0
        text = False

    class cntr:
        """Controls for program

        next_k - next image
        prev_k - prev image
        save - save single image (in mode)
        save_all - save all images (in mode)
        exit_k - exit the program

        dice - calculate dice value
        dice_more - show all dice values based on dataset

        m1_k etc. - mode selection

        modes [dict] dictionary with mode names
        """
        next_k = ord('m')
        prev_k = ord('n')
        save = ord('s')
        save_all = ord('z')
        exit_k = 27

        dice = ord('d')
        dice_more = ord('f')

        m1_k = ord('1')
        m2_k = ord('2')
        m3_k = ord('3')
        m4_k = ord('4')
        m5_k = ord('5')

        modes = {
            0: 'original',
            1: 'hsv_filter',
            2: 'ws_mask',
            3: 'ws_mask_bg',
            4: 'fgbg_segm',
            5: 'ws_fgbg_segm'
        }

    class xtra:
        """Ends and odds

        disco [bool] random colors for masks on each loop?
        show_save_all [bool] run saving all in foreground?
        """
        disco = False
        show_save_all = True
