import cv2 as cv
import numpy as np
import os
import glob

from stereo_vision.constants.cv_window_names import LEFT_CAM_WINDOW, RIGHT_CAM_WINDOW


class CaptureFrames:
    def __init__(self):
        self.left_cam = "/dev/video2"
        self.right_cam = "/dev/video4"
        self.left_cap = cv.VideoCapture()
        self.right_cap = cv.VideoCapture()

        self.left_cap.open(self.left_cam, cv.CAP_ANY)
        if not self.left_cap.isOpened():
            raise Exception(
                'Camera', 'left camera not found')
        self.right_cap.open(self.right_cam, cv.CAP_ANY)
        if not self.right_cap.isOpened():
            raise Exception(
                'Camera', 'right camera not found'
            )
        self.capture_frames()
        pass

    def get_file_name(self, frame_count, isLeft=False):
        folder = ''
        if isLeft:
            folder = 'left'
        else:
            folder = 'right'
        fileName = 'frame_' + str(frame_count) + '.jpg'
        return 'images/calibration/{}/{}'.format(folder, fileName)

    def capture_frames(self):
        print('Press S or s to save calibration')
        frame_count = 0
        while True:
            _, l_frame = self.left_cap.read()
            _, r_frame = self.right_cap.read()
            if l_frame is None or r_frame is None:
                print('Error! no left or right frame')
                raise Exception('Camera', 'Frame not found')

            cv.imshow(LEFT_CAM_WINDOW, l_frame)
            cv.imshow(RIGHT_CAM_WINDOW, r_frame)

            iKey = cv.waitKey(5)  # Wait for key press
            if iKey == ord('s') or iKey == ord('S'):
                l_filename = self.get_file_name(
                    frame_count, isLeft=True)
                r_filename = self.get_file_name(frame_count, isLeft=False)
                cv.imwrite(l_filename, l_frame)
                cv.imwrite(r_filename, r_frame)
                frame_count += 1
                print('Left frame: {}, Right frame: {} are created.'.format(
                    l_filename, r_filename))
            elif iKey == ord('q') or iKey == ord('Q'):
                break
