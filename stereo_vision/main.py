
import cv2 as cv
import argparse
import sys

def initialize_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", help="Start calibration for stereo camera")
    return parser;

def calibrate_stereo_cam():
    print('handle stereo cam')

def handle_arguments(parser):
    args = parser.parse_args()
    if args.calibrate:
        calibrate_stereo_cam()


def main():
    parser = initialize_args()
    handle_arguments(parser)
    print('hello world')