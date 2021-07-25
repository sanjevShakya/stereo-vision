from os import write
import cv2 as cv
import time 
import matplotlib.pyplot as plt
import numpy as np

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def calc_disparity_map(left_img, right_img):
    window_size = 5
    min_disp = 0
    num_disp = 112 # odd number
    # stereo = cv.StereoBM_create(
    #     numDisparities=16,
    #     blockSize=15
    # )
    stereo = cv.StereoSGBM_create(
        minDisparity=16,
        blockSize=5,
        preFilterCap=6,
        numDisparities=16,
        uniquenessRatio=2,
        speckleWindowSize=0,
        disp12MaxDiff=0,
        P1=8*3*window_size**2,
        P2=32*3*window_size**2,
    )
    disparity = stereo.compute(left_img, right_img)
    return disparity


def remove_invalid(disp_arr, points, colors):
    mask = (
        (disp_arr > disp_arr.min()) &
        np.all(~np.isnan(points), axis=1) &
        np.all(~np.isinf(points), axis=1)
    )
    return points[mask], colors[mask]


def calc_point_cloud(image, disp, q):
    points = cv.reprojectImageTo3D(disp, q).reshape(-1, 3)
    colors = image.reshape(-1, 3)
    return remove_invalid(disp.reshape(-1), points, colors)


def project_points(points, colors, r, t, k, dist_coeff, width, height):
    projected, _ = cv.projectPoints(points, r, t, k, dist_coeff)
    xy = projected.reshape(-1, 2).astype(np.int)
    mask = (
        (0 <= xy[:, 0]) & (xy[:, 0] < width) &
        (0 <= xy[:, 1]) & (xy[:, 1] < height)
    )
    return xy[mask], colors[mask]


def calc_projected_image(points, colors, r, t, k, dist_coeff, width, height):
    xy, cm = project_points(points, colors, r, t, k, dist_coeff, width, height)
    image = np.zeros((height, width, 3), dtype=colors.dtype)
    image[xy[:, 1], xy[:, 0]] = cm
    return image


def write_ply(file_name, verts, colors):
    ''' function to write a ply file '''

    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])

    verts = verts[~np.isnan(verts).any(axis=1)]
    verts = verts[~np.isinf(verts).any(axis=1)]

    with open(file_name, 'w') as file:
        file.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(file, verts, '%f %f %f %d %d %d')

def create_output(filename, vertices, colors):
    colors = colors.reshape(-1,3)
    vertices = np.hstack([vertices.reshape(-1,3),colors])
    vertices = vertices[~np.isnan(vertices).any(axis=1)]
    vertices = vertices[~np.isinf(vertices).any(axis=1)]
    
    with open(filename, 'w') as file:
        file.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(file,vertices,'%f %f %f %d %d %d')

class DisparityReconstruction():
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

    def compute(self, left_image, right_image):
        
        cv.imshow('left', left_image)
        cv.waitKey(1000)
        cv.imshow('right', right_image)
        cv.waitKey(1000)

        image = right_image
        height, width, _ = image.shape

        R1, R2, P1, P2, Q, roi_left, roi_right = cv.stereoRectify(
            self.mtx1, self.dist1, self.mtx2, self.dist2, (width, height), self.R, self.T)
        print('stereo Rectify complete', Q)
        gray_left_image = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
        gray_right_image = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)
        start_time = time.time()
        disparity_map = calc_disparity_map(gray_left_image, gray_right_image)
        print("time: {} s".format(time.time() - start_time))
        plt.imshow(disparity_map, 'viridis')
        plt.savefig('disparity_map.png')
        plt.show()
        print('Disparity map computed', disparity_map)
        points = cv.reprojectImageTo3D(disparity_map, Q)
        # colors = cv.imread(right_image, cv.COLOR_RGB2BGR)
        colors = cv.cvtColor(left_image, cv.COLOR_BGR2RGB)
        mask  = disparity_map > disparity_map.min()
        out_points = points[mask]
        out_colors = colors[mask]
    
        create_output('output/disparity_reconstruction.ply', out_points, out_colors)

        # save z_norm as an image (change [0,1] range to [0,255] range with uint8 type)
        # img = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
        # o3d.io.write_image("../../TestData/sync.png", img)
        # o3d.visualization.draw_geometries([img])
        # points, colors = calc_point_cloud(right_image, disparity_map, Q)
