
import cv2 as cv
import argparse
import sys

from stereo_vision.util.capture_frames import CaptureFrames
from stereo_vision.util.stereo_calibration import StereoCalibration


def initialize_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibrate", help="Start calibration for stereo camera")
    parser.add_argument(
        "--captureframes", help="Start capturing frame from stereo camera"
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


def handle_arguments(parser):
    args = parser.parse_args()
    if args.calibrate:
        calibrate_stereo_cam()
    if args.captureframes:
        capture_frames()


def main():
    parser = initialize_args()
    handle_arguments(parser)
