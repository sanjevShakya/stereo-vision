
import cv2 as cv
import argparse
import sys

from stereo_vision.util.capture_frames import CaptureFrames
from stereo_vision.util.stereo_calibration import StereoCalibration
from stereo_vision.util.reconstruction import Reconstruction
from stereo_vision.util.cv_file_storage import CVFileUtil
from stereo_vision.constants.file_path import CALIB_FILE_PATH
from stereo_vision.util.disparity3dreconstruction import DisparityReconstruction


def initialize_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibrate", help="Start calibration for stereo camera")
    parser.add_argument(
        "--captureframes", help="Start capturing frame from stereo camera"
    )
    parser.add_argument(
        '--reconstruct', help="start computing 3d matches and drawing correspondences")
    parser.add_argument(
        '--disparityre', help="disparity reconstruction"
    )
    return parser


def capture_frames():
    capFrames = CaptureFrames()


def calibrate_stereo_cam():
    print('handle stereo cam')
    stereoCalibration = StereoCalibration()
    stereoCalibration.read_images()
    print('Camera model:: ', stereoCalibration.camera_model)
    print("Stereo camera calibration finished")


def get_camera_model():
    fileUtil = CVFileUtil(CALIB_FILE_PATH)
    fileUtil.open_read_mode()
    camera_model = {}
    camera_model_keys = ['mtx1', 'mtx2', 'dist1',
                         'dist2', 'rvecs1', 'rvecs2', 'R', 'T', 'E', 'F']
    for key in camera_model_keys:
        camera_model[key] = fileUtil.read_matrix(key)
        # fileUtil.write_matrix(key, self.camera_model[key])
    fileUtil.close_file()

    return camera_model


def start_reconstruction():
    camera_model = get_camera_model()
    image_frame1_path = 'images/calibration/left/frame_{}.jpg'.format(0)
    image_frame2_path = 'images/calibration/right/frame_{}.jpg'.format(0)
    frame1 = cv.imread(image_frame1_path)
    frame2 = cv.imread(image_frame2_path)
    cv.namedWindow(image_frame1_path, flags=cv.WINDOW_GUI_EXPANDED)
    cv.namedWindow(image_frame2_path, flags=cv.WINDOW_GUI_EXPANDED)
    reconstruction = Reconstruction(camera_model['mtx1'], camera_model['mtx2'], camera_model['dist1'], camera_model['dist2'],
                                    camera_model['rvecs1'], camera_model['rvecs2'], camera_model['R'], camera_model['T'], camera_model['E'], camera_model['F'])

    reconstruction.feature_match(frame1, frame2)


def disparity_reconstruction():
    camera_model = get_camera_model()
    re = DisparityReconstruction(camera_model['mtx1'], camera_model['mtx2'], camera_model['dist1'], camera_model['dist2'],
                                 camera_model['rvecs1'], camera_model['rvecs2'], camera_model['R'], camera_model['T'], camera_model['E'], camera_model['F'])
    image_frame1_path = 'images/calibration/left/frame_{}.jpg'.format(0)
    image_frame2_path = 'images/calibration/right/frame_{}.jpg'.format(0)
    print(image_frame1_path)
    print(image_frame2_path)
    frame1 = cv.imread(image_frame1_path)
    frame2 = cv.imread(image_frame2_path)
    print(frame1)
    print(frame2)
    re.compute(frame1, frame2)


def handle_arguments(parser):
    args = parser.parse_args()
    if args.calibrate:
        calibrate_stereo_cam()
    if args.captureframes:
        capture_frames()
    if args.reconstruct:
        start_reconstruction()
    if args.disparityre:
        disparity_reconstruction()


def main():
    parser = initialize_args()
    handle_arguments(parser)
