import numpy as np
import cv2 as cv
import glob

RIGHT_CAM_DIR = 'right'
LEFT_CAM_DIR = 'left'


class StereoCalibration():
    left_image_points = []
    right_image_points = []
    three_d_points = []
    image_shape = None
    camera_model = None
    termination_criteria = (cv.TERM_CRITERIA_EPS +
                            cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_criteria = (cv.TERM_CRITERIA_EPS +
                            cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    real_world_points = []  # 3d point in real world space
    l_img_points = []  # 2d points in image plane.
    r_img_points = []  # 2d points in image plane.

    def __init__(self, calibration_dir, file_out_path, filename='stereo_calibration.yaml', row_pattern=9, col_pattern=6):
        self.calibration_dir = calibration_dir
        self.outdir = file_out_path
        self.filename = filename
        self.pattern = (row_pattern, col_pattern)
        pass

    def getImagePath(self, subdir):
        return self.calibration_dir + subdir + '/*.jpg'

    def read_images(self):
        right_images = glob.glob(self.getImagePath(RIGHT_CAM_DIR))
        left_images = glob.glob(self.getImagePath(LEFT_CAM_DIR))

        right_images.sort()
        left_images.sort()
        if len(right_images) != len(left_images):
            raise Exception(
                'Calibration', 'The number of images for both left and right camera should be same.')
        total_images = len(right_images)

        for i in range(total_images):
            l_img = cv.imread(left_images[i])
            r_img = cv.imread(right_images[i])

            gray_l_img = cv.cvtColor(l_img, cv.COLOR_BGR2GRAY)
            gray_r_img = cv.cvtColor(r_img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret_l, corners_l = cv.findChessboardCorners(
                gray_l_img, self.pattern, None)
            ret_r, corners_r = cv.findChessboardCorners(
                gray_r_img, self.pattern, None)

            self.real_world_points.append(self.objp)

            if ret_l and ret_r:
                cv.cornerSubPix(gray_l_img, corners_l, (11, 11),
                                (-1, -1), self.calibration_criteria)
                cv.cornerSubPix(gray_r_img, corners_r, (11, 11),
                                (-1, -1), self.calibration_criteria)

                self.left_image_points.append(corners_l)
                self.right_image_points.append(corners_r)

                ret_l = cv.drawChessboardCorners(
                    l_img, self.pattern, corners_l, ret_l)
                ret_r = cv.drawChessboardCorners(
                    r_img, self.pattern, corners_r, ret_r)

                cv.imshow(left_images[i], l_img)
                cv.waitKey(500)

                cv.imshow(right_images[i], r_img)
                cv.waitKey(500)

            self.image_shape = gray_l_img.shape[::-1]

            _, self.M1, self.d1, self.r1, self.t1 = cv.calibrateCamera(
                self.real_world_points, self.l_img_points, self.image_shape, None, None)

            _, self.M2, self.d2, self.r2, self.t2 = cv.calibrateCamera(
                self.real_world_points, self.r_img_points, self.image_shape, None, None)

            self.camera_model = self.stereo_calibrate()

    def stereo_calibrate(self):
        flags = 0
        flags |= cv.CALIB_FIX_INTRINSIC
        flags |= cv.CALIB_USE_INTRINSIC_GUESS
        flags |= cv.CALIB_FIX_FOCAL_LENGTH
        flags |= cv.CALIB_ZERO_TANGENT_DIST

        stereocalib_criteria = (cv.TERM_CRITERIA_MAX_ITER +
                                cv.TERM_CRITERIA_EPS, 100, 1e-5)
        _, mtx1, dist1, mtx2, dist2, R, T, E, F = cv.stereoCalibrate(
            self.real_world_points, self.l_img_points,
            self.r_img_points, self.mtx1, self.dist1, self.mtx2,
            self.dist2, self.image_shape,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', mtx1)
        print('Distortion1 ', dist1)
        print('Intrinsic_mtx_2', mtx2)
        print('distortion 2', dist2)
        print('Rotation:', R)
        print('Translation:', T)
        print('Essential Matrix', E)
        print('Fundamental Matrix', F)

        camera_model = dict([('mtx1', mtx1), ('mtx2', mtx2), ('dist1', dist1),
                            ('dist2', dist2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

        cv.destroyAllWindows()
        return camera_model
