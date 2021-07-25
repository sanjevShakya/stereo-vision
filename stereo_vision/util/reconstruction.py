import cv2 as cv
import numpy as np
import time

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header
'''


def create_output(filename, vertices):
    # vertices = vertices[~np.isnan(vertices).any(axis=1)]
    # vertices = vertices[~np.isinf(vertices).any(axis=1)]
    print(vertices)
    with open(filename, 'w') as file:
        file.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(file, vertices, '%f %f %f')


class Reconstruction:
    def __init__(self, mtx1, mtx2, dist1, dist2, rvecs1, rvecs2, R, T, E, F):
        self.mtx1 = mtx1
        self.mtx2 = mtx2
        self.dist1 = dist1
        self.dist2 = dist2
        self.rvecs1 = rvecs1
        self.rvecs2 = rvecs2
        self.R = R
        self.T = T
        self.E = E
        self.F = F
        self.probInlier = 0.8
        self.matcher = cv.DescriptorMatcher_create("BruteForce-Hamming")
        self.detector_name = 'akaze'
        if self.detector_name == 'orb':
            self.detector = cv.ORB_create()
        if self.detector_name == 'akaze':
            self.detector = cv.AKAZE_create()
            self.detector.setThreshold(3e-4)
        if self.detector_name == 'sift':
            self.detector = cv.SIFT_create()
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)
            self.matcher = cv.FlannBasedMatcher(index_params,search_params)


    def detect_keypoints(self, frame, camera_mtx, distortion_coeff):
        c_frame = frame.copy()
        kps = self.detector.detect(c_frame, None)
        kps, desc = self.detector.compute(c_frame, kps)
        return kps, desc

    def get_matches(self, frame1_desc, frame2_desc):
        return self.matcher.knnMatch(frame1_desc, frame2_desc, k=2)

    def compute_inliers(self, matches, frame1_kps, frame2_kps):
        good_matches = [m for i, (m, n) in enumerate(
            matches) if m.distance < self.probInlier * n.distance]
        frame1_matches = np.float32(
            [frame1_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        frame2_matches = np.float32(
            [frame2_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        return (good_matches, frame1_matches, frame2_matches)

    def compute_fundamental_matrix(self, frame1_matches, frame2_matches):
        essential_mtx, mask = cv.findEssentialMat(
            frame1_matches, frame2_matches, self.mtx1, method=cv.RANSAC, threshold=0.3)
        essential_mtx = np.array(essential_mtx)
        F, mask = cv.findFundamentalMat(
            frame1_matches, frame2_matches, cv.FM_LMEDS)

        fundamental_mtx_proj = self.compute_stereo_fundamental_mtx(
            essential_mtx)

        # print('F shape', F)
        print('F from calibration', self.F)
        print('fundamental_mtx_proj', fundamental_mtx_proj)
        return F, mask

    def compute_stereo_fundamental_mtx(self, essential_mtx):
        P_left = self.compute_projection_mtx(self.mtx1, self.rvecs1, self.T)
        P_right = self.compute_projection_mtx(self.mtx2, self.rvecs2, self.T)
        P_Pinv = np.matmul(P_left, np.linalg.pinv(P_right))
        return essential_mtx * P_Pinv

    def compute_fundamental_mtx_from_projection(self, essential_mtx, P, Pprime):
        P_Pprime = np.matmul(P.T, np.linalg.pinv(Pprime))
        return essential_mtx * P_Pprime

    def compute_projection_mtx(self, mtx, rvecs, tvecs):
        r_mat = np.zeros(shape=(3, 3))
        R = np.array(cv.Rodrigues(rvecs[0])[0])
        temp = np.array(np.matmul(mtx, R))
        P = np.hstack((temp, tvecs))
        return P

    def epipolar_line(self, x, x_prime, F):
        x = np.array(x).T
        x = np.vstack((x, [1]))
        x_prime = np.array(x_prime).T
        x_prime = np.vstack((x_prime, [1]))
        frame2_l = np.dot(F, x)
        frame1_l = np.dot(F.T, x_prime)

        frame1_l = frame1_l / frame1_l[2]
        frame2_l = frame2_l / frame2_l[2]

        return (frame1_l, frame2_l)

    def line_x(self, homogeneous_coordinate, x):
        a = homogeneous_coordinate[0]
        b = homogeneous_coordinate[1]
        c = homogeneous_coordinate[2]
        return (-(a * x + c)) / b

    def line_y(self, homogeneous_coordinate, y):
        a = homogeneous_coordinate[0]
        b = homogeneous_coordinate[1]
        c = homogeneous_coordinate[2]
        return (-(b * y + c)) / a

    def draw_line_frame(self, window_name, frame, line, height):
        x1 = 0
        y1 = int(self.line_x(line, x1)[0])
        y2 = height
        x2 = int(self.line_y(line, y2)[0])
        cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.imshow(window_name, frame)

    def feature_match(self, frame1, frame2):
        frame1 = cv.undistort(frame1, self.mtx1, self.dist1)
        frame2 = cv.undistort(frame2, self.mtx2, self.dist2)
        # cv.imshow('frame1', frame1)
        # cv.waitKey(500)
        # cv.imshow('frame2', frame2)
        # cv.waitKey(500)
        start_time = time.time()
        frame1_kps, frame1_desc = self.detect_keypoints(
            frame1, self.mtx1, self.dist1)
        frame2_kps, frame2_desc = self.detect_keypoints(
            frame2, self.mtx2, self.dist2)
        matches = self.get_matches(frame1_desc, frame2_desc)
        good_matches, frame1_matches, frame2_matches = self.compute_inliers(
            matches, frame1_kps, frame2_kps)
        F, mask = self.compute_fundamental_matrix(frame1_matches, frame2_matches)
        print("time: {} s".format(time.time() - start_time))
        # essential_mtx = cv.findEssentialMat(
        #     frame1_matches, frame2_matches, self.mtx1, method=cv.RANSAC, prob=0.8, threshold=0.2)
        height, width, _ = frame1.shape
        res = cv.drawMatches(frame1, frame1_kps, frame2, frame2_kps,
                         good_matches, None, matchColor=(255, 255, 0), singlePointColor=(255, 255, 0))
        cv.imwrite("frame_{}_detector.jpg".format(self.detector_name), res)
        cv.imshow('another_window', res)
        # cv.waitKey(2000)
        R1, R2, P1, P2, Q, roi_left, roi_right = cv.stereoRectify(
            self.mtx1, self.dist1, self.mtx2, self.dist2, (width, height), self.R, self.T)
        # R1, R2, t = cv.decomposeEssentialMat(essential_mtx)
        # P1 = np.hstack((R1, t))
        # P2 = np.hstack((R2, t))
        height, width, _ = frame1.shape
        print('height', height)
        print('width', width)
        print(frame1_matches)
        pts1 = np.int32(frame1_matches)
        pts2 = np.int32(frame2_matches)
        pts1 = frame1_matches[mask.ravel()==1]
        pts2 = frame2_matches[mask.ravel()==1]
        print('here')

        # pp_1 = []
        # pp_2 = []
        # indices = []
        # # print('inliers_frame1 0', inliers_frame1[0])
        # for i in range(len(frame1_matches)):
        #     p1 = frame1_matches[i][0]
        #     p2 = frame2_matches[i][0]
        #     pp_1.append([p1[0], p1[1], 1.0])
        #     pp_2.append([p2[0], p2[1], 1.0])
        # pp_1 = np.array(pp_1).T
        # pp_2 = np.array(pp_2).T
        # pts1 = np.dot(np.linalg.pinv(P1), pp_1)
        # pts2 = np.dot(np.linalg.pinv(P2), pp_2)

        three_d_points = cv.triangulatePoints(
            P1, P2, pts1, pts2)
        three_d_points /= three_d_points[3]
        three_d_points = three_d_points[:-1].T
        create_output('output/traingulated_contruction.ply', three_d_points)
        # new_F = self.computeF(frame1_matches[0], frame2_matches[0])
        # print('new_f', new_F)
        # for i in range(len(frame1_matches)):
        #     x = frame1_matches[i]
        #     xp = frame2_matches[i]
        #     frame1_l, frame2_l = self.epipolar_line(x, xp, F)
        #     self.draw_line_frame('epipole_frame1', frame1,
        #                          frame1_l, height)

        #     self.draw_line_frame('epipole_frame2', frame2,
        #                          frame2_l, height)
    # def pre_process_input(self, left_image, right_image):
    #     left_kps, left_desc = left_image
    #     right_kps, right_desc = right_image
