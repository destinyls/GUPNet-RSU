import numpy as np
import cv2
import pdb
import os
import math
################  Object3D  ##################

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects

def draw_box_3d(image, corners, c=(0, 255, 0)):
    face_idx = [[0,1,5,4],
                [1,2,6,5],
                [2,3,7,6],
                [3,0,4,7]]
    points = []
    for ind_f in [3, 2, 1, 0]:
        f = face_idx[ind_f]
        for j in [0, 1, 2, 3]:
            cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
                    (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), c, 2, lineType=cv2.LINE_AA)
            if [int(corners[f[j], 0]), int(corners[f[j], 1])] not in points:
                points.append([int(corners[f[j], 0]), int(corners[f[j], 1])])
            if [int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])] not in points:
                points.append([int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])])

        if ind_f == 0:
            cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
                    (int(corners[f[2], 0]), int(corners[f[2], 1])), c, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
                    (int(corners[f[3], 0]), int(corners[f[3], 1])), c, 1, lineType=cv2.LINE_AA)
        
    if len(points) == 8:
        points_1 = points[:4]
        points_2 = [points[7], points[6], points[4], points[5]]
        points_3 = [points[0], points[3], points[5], points[4]]
        points_4 = [points[1], points[2], points[7], points[6]]
        points_5 = [points[2], points[3], points[5], points[7]]
        points_6 = [points[1], points[0], points[4], points[6]]

        points_1 = np.array([points_1])
        points_2 = np.array([points_2])
        points_3 = np.array([points_3])
        points_4 = np.array([points_4])
        points_5 = np.array([points_5])
        points_6 = np.array([points_6])
        zeros = np.zeros((image.shape), dtype=np.uint8)
        if c == (0, 255, 0):
            c = (128, 205, 67)
        elif c == (255, 0, 0):
            c = (237, 149, 100)
        mask = cv2.fillPoly(zeros, points_1, color=c)
        mask = cv2.fillPoly(mask, points_2, color=c)
        mask = cv2.fillPoly(mask, points_3, color=c)
        mask = cv2.fillPoly(mask, points_4, color=c)
        mask = cv2.fillPoly(mask, points_5, color=c)
        mask = cv2.fillPoly(mask, points_6, color=c)
        image = 0.3 * mask + image
    return image

class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.trucation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.pos = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()

    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if self.trucation == -1:
            self.level_str = 'DontCare'
            return 0

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4


    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos
        return corners3d

    def generate_corners3d_denorm(self, denorm):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.dot(R, corners3d)

        # ground palne equation
        denorm = denorm[:3]
        denorm_norm = denorm / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
        ori_denorm = np.array([0.0, -1.0, 0.0])
        theta = -1 * math.acos(np.dot(denorm_norm, ori_denorm))
        n_vector = np.cross(denorm, ori_denorm)
        n_vector_norm = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
        rotation_matrix, j = cv2.Rodrigues(theta * n_vector_norm)
        corners3d = np.dot(rotation_matrix, corners3d)
        corners3d = corners3d.T + self.pos

        center3d = np.mean(corners3d, axis=0)
        return center3d, corners3d

    def to_bev_box2d(self, oblique=True, voxel_size=0.1):
        """
        :param bev_shape: (2) for bev shape (h, w), => (y_max, x_max) in image
        :param voxel_size: float, 0.1m
        :param oblique:
        :return: box2d (4, 2)/ (4) in image coordinate
        """
        if oblique:
            corners3d = self.generate_corners3d()
            xz_corners = corners3d[0:4, [0, 2]]
            box2d = np.zeros((4, 2), dtype=np.int32)
            box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            box2d[:, 1] = Object3d.BEV_SHAPE[0] - 1 - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            box2d[:, 0] = np.clip(box2d[:, 0], 0, Object3d.BEV_SHAPE[1])
            box2d[:, 1] = np.clip(box2d[:, 1], 0, Object3d.BEV_SHAPE[0])
        else:
            box2d = np.zeros(4, dtype=np.int32)
            # discrete_center = np.floor((self.pos / voxel_size)).astype(np.int32)
            cu = np.floor((self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            cv = Object3d.BEV_SHAPE[0] - 1 - ((self.pos[2] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            half_l, half_w = int(self.l / voxel_size / 2), int(self.w / voxel_size / 2)
            box2d[0], box2d[1] = cu - half_l, cv - half_w
            box2d[2], box2d[3] = cu + half_l, cv + half_w

        return box2d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.pos, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.trucation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.pos[0], self.pos[1], self.pos[2],
                       self.ry)
        return kitti_str

###################  calibration  ###################
def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


class Calibration(object):
    def __init__(self, calib_file):
        if isinstance(calib_file, str):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4
        self.C2V = self.inverse_rigid_trans(self.V2C)

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_lidar(self, pts_rect):
        pts_ref = np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_rect)))
        pts_ref = self.cart_to_hom(pts_ref)  # nx4
        return np.dot(pts_ref, np.transpose(self.C2V))

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def depthmap_to_rect(self, depth_map):
        """
        :param depth_map: (H, W), depth_map
        :return:
        """
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        """
        Can only process valid u, v, d, which means u, v can not beyond the image shape, reprojection error 0.02
        :param u: (N)
        :param v: (N)
        :param d: (N), the distance between camera and 3d points, d^2 = x^2 + y^2 + z^2
        :return:
        """
        assert self.fu == self.fv, '%.8f != %.8f' % (self.fu, self.fv)
        fd = np.sqrt((u - self.cu) ** 2 + (v - self.cv) ** 2 + self.fu ** 2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d ** 2 - x ** 2 - y ** 2)
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
        return pts_rect

    def inverse_rigid_trans(self, Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    def roty2alpha(self, ry, pos):
        alpha = ry - np.arctan2(pos[0], pos[2])
        if alpha > np.pi:
            alpha -= 2 * np.pi
        if alpha < -np.pi:
            alpha += 2 * np.pi
        return alpha

    def alpha2roty(self, alpha, pos):
        ry = alpha + np.arctan2(pos[0], pos[2])
        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi
        return ry

    def alpha2ry(self, alpha, u):
        """
        Get rotation_y by alpha + theta - 180
        alpha : Observation angle of object, ranging [-pi..pi]
        x : Object center x to the camera center (x-W/2), in pixels
        rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
        """
        ry = alpha + np.arctan2(u - self.cu, self.fu)
        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi
        return ry

    def ry2alpha(self, ry, u):
        alpha = ry - np.arctan2(u - self.cu, self.fu)
        if alpha > np.pi:
            alpha -= 2 * np.pi
        if alpha < -np.pi:
            alpha += 2 * np.pi

        return alpha
    def flip(self,img_size):      
        wsize = 4
        hsize = 2
        p2ds = (np.concatenate([np.expand_dims(np.tile(np.expand_dims(np.linspace(0,img_size[0],wsize),0),[hsize,1]),-1),\
                                np.expand_dims(np.tile(np.expand_dims(np.linspace(0,img_size[1],hsize),1),[1,wsize]),-1),
                                np.linspace(2,78,wsize*hsize).reshape(hsize,wsize,1)],-1)).reshape(-1,3)
        p3ds = self.img_to_rect(p2ds[:,0:1],p2ds[:,1:2],p2ds[:,2:3])
        p3ds[:,0]*=-1
        p2ds[:,0] = img_size[0] - p2ds[:,0]
        
        #self.P2[0,3] *= -1
        cos_matrix = np.zeros([wsize*hsize,2,7])
        cos_matrix[:,0,0] = p3ds[:,0]
        cos_matrix[:,0,1] = cos_matrix[:,1,2] = p3ds[:,2]
        cos_matrix[:,1,0] = p3ds[:,1]
        cos_matrix[:,0,3] = cos_matrix[:,1,4] = 1
        cos_matrix[:,:,-2] = -p2ds[:,:2]
        cos_matrix[:,:,-1] = (-p2ds[:,:2]*p3ds[:,2:3])
        new_calib = np.linalg.svd(cos_matrix.reshape(-1,7))[-1][-1]
        new_calib /= new_calib[-1]
        
        new_calib_matrix = np.zeros([4,3]).astype(np.float32)
        new_calib_matrix[0,0] = new_calib_matrix[1,1] = new_calib[0]
        new_calib_matrix[2,0:2] = new_calib[1:3]
        new_calib_matrix[3,:] = new_calib[3:6]
        new_calib_matrix[-1,-1] = self.P2[-1,-1]
        self.P2 = new_calib_matrix.T
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv) 
        
    def affine_transform(self,img_size,trans):
        wsize = 4
        hsize = 2
        random_depth = np.linspace(2,78,wsize*hsize).reshape(hsize,wsize,1)
        p2ds = (np.concatenate([np.expand_dims(np.tile(np.expand_dims(np.linspace(0,img_size[0],wsize),0),[hsize,1]),-1),np.expand_dims(np.tile(np.expand_dims(np.linspace(0,img_size[1],hsize),1),[1,wsize]),-1),random_depth],-1)).reshape(-1,3)
        p3ds = self.img_to_rect(p2ds[:,0:1],p2ds[:,1:2],p2ds[:,2:3])
        p2ds[:,:2] = np.dot(np.concatenate([p2ds[:,:2],np.ones([wsize*hsize,1])],-1),trans.T)

        cos_matrix = np.zeros([wsize*hsize,2,7])
        cos_matrix[:,0,0] = p3ds[:,0]
        cos_matrix[:,0,1] = cos_matrix[:,1,2] = p3ds[:,2]
        cos_matrix[:,1,0] = p3ds[:,1]
        cos_matrix[:,0,3] = cos_matrix[:,1,4] = 1
        cos_matrix[:,:,-2] = -p2ds[:,:2]
        cos_matrix[:,:,-1] = (-p2ds[:,:2]*p3ds[:,2:3])
        new_calib = np.linalg.svd(cos_matrix.reshape(-1,7))[-1][-1]
        new_calib /= new_calib[-1]
        
        new_calib_matrix = np.zeros([4,3]).astype(np.float32)
        new_calib_matrix[0,0] = new_calib_matrix[1,1] = new_calib[0]
        new_calib_matrix[2,0:2] = new_calib[1:3]
        new_calib_matrix[3,:] = new_calib[3:6]
        new_calib_matrix[-1,-1] = self.P2[-1,-1]
        return new_calib_matrix.T

class PointCloudFilter(object):
    """
    Class for getting lidar-bev ground truth annotation from camera-view
    annotation.
    :param res: float. resolution in meters. Each output pixel will
                represent an square region res x res in size.
    :param side_range: tuple of two floats. (left-most, right_most)
    :param fwd_range: tuple of two floats. (back-most, forward-most)
    :param height_range: tuple of two floats. (min, max)
    :param calib: class instance for getting transform matrix from
                  calibration.
    """
    def __init__(self,
                 side_range=(-39.68, 39.68),
                 fwd_range=(0, 69.12),
                 height_range=(-2., -2.),
                 res=0.10):
        self.res = res
        self.side_range = side_range
        self.fwd_range = fwd_range
        self.height_range = height_range

    def set_range_patameters(self, side_range, fwd_range, height_range):
        self.side_range = side_range
        self.fwd_range = fwd_range
        self.height_range = height_range

    def read_bin(self, path):
        """
        Helper function to read one frame of lidar pointcloud in .bin format.
        :param path: where pointcloud is stored in .bin format.
        :return: (x, y, z, intensity) of pointcloud, N x 4.
        """
        points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4])
        x_points, y_points, z_points, indices = self.get_pcl_range(points)
        filtered_points = np.concatenate((x_points[:,np.newaxis], y_points[:,np.newaxis], z_points[:,np.newaxis]), axis = 1)
        return filtered_points

    def scale_to_255(self, value, minimum, maximum, dtype=np.uint8):
        """
        Scales an array of values from specified min, max range to 0-255.
        Optionally specify the data type of the output (default is uint8).
        """
        if minimum!= maximum:
            return (((value - minimum) / float(maximum - minimum))
                    * 255).astype(dtype)
        else:
            return self.get_meshgrid()

    def get_pcl_range(self, points):
        """
        Get the pointcloud wihtin side_range and fwd_range.
        :param points: np.float, N x 4. each column is [x, y, z, intensity].
        :return: [x, y, z, intensity] of filtered points and corresponding
                 indices.
        """
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]
        indices = []
        for i in range(points.shape[0]):
            if points[i, 0] > self.fwd_range[0] and points[i, 0] < self.fwd_range[1]:
                if points[i, 1]  > self.side_range[0] and points[i, 1] < self.side_range[1]:
                    indices.append(i)

        indices = np.array(indices)
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        x_points = x_points[indices]
        y_points = y_points[indices]
        z_points = z_points[indices]
        return x_points, y_points, z_points, indices

    def clip_height(self, z_points):
        """
        Clip the height between (min, max).
        :param z_points: z_points from get_pcl_range
        :return: clipped height between (min,max).
        """
        height = np.clip(
            a=z_points, a_max=self.height_range[1], a_min=self.height_range[0]
        )
        return height

    def get_meshgrid(self):
        """
        Create mesh grids (size: res x res) in the x-y plane of the lidar
        :return: np.array: uint8, x-y plane mesh grids based on resolution.
        """
        x_max = 1 + int((self.side_range[1] - self.side_range[0]) / self.res)
        y_max = 1 + int((self.fwd_range[1] - self.fwd_range[0]) / self.res)
        img = np.zeros([y_max, x_max], dtype=np.uint8)
        return img

    def pcl2xy_plane(self, x_points, y_points):
        """
        Convert the lidar coordinate to x-y plane coordinate.
        :param x_points: x of points in lidar coordinate.
        :param y_points: y of points in lidar coordinate.
        :return: corresponding pixel position based on resolution.
        """
        x_img = (-y_points / self.res).astype(np.int32) # x axis is -y in lidar
        y_img = (-x_points / self.res).astype(np.int32) # y axis is -x in lidar
        # shift pixels to have minimum be (0,0)
        x_img -= int(np.floor(self.side_range[0] / self.res))
        y_img += int(np.ceil(self.fwd_range[1] / self.res))
        return x_img, y_img

    def pcl_2_bev(self, points):
        """
        Creates an 2D birds eye view representation of the pointcloud.
        :param points: np.float, N x 4. input pointcloud matrix,
                       each column is [x, y, z, intensity]
        :return: np.array, representing an image of the BEV.
        """
        # rescale the height values - to be between the range 0-255
        x_points, y_points, z_points, _ = self.get_pcl_range(points)
        x_img, y_img = self.pcl2xy_plane(x_points, y_points)
        height = self.clip_height(z_points)
        bev_img = self.get_meshgrid()
        pixel_values = self.scale_to_255(height,
                                         self.height_range[0],
                                         self.height_range[1])
        # fill pixel values in image array
        x_img = np.clip(x_img, 0, bev_img.shape[1] - 1)
        y_img = np.clip(y_img, 0, bev_img.shape[0] - 1)
        print(bev_img.shape)
        bev_img[y_img, x_img] = 255
        return bev_img

    def pcl_anno_bev(self, points, point_road_index, point_out_road_index):
        """
        :param points: np.float, N x 4. input pointcloud matrix.
        :param point_road_index: index of points on free road surface.
        :param point_out_road_index: index of points out free road surface.
        :return: BEV of lidar pointcloud containing free road annotation. 
        """
        x_points, y_points, _, _, indices = self.get_pcl_range(points)
        on_road = []
        out_road = []
        for i in range(len(indices)):
            if indices[i] in point_road_index:
                on_road.append(i)
            if indices[i] in point_out_road_index:
                out_road.append(i)
        pcl_anno_bev_img = self.get_meshgrid()
        x_points_road = x_points[on_road]
        y_points_road = y_points[on_road]
        x_road_img, y_road_img = self.pcl2xy_plane(x_points_road,
                                                   y_points_road)
        x_points_out_road = x_points[out_road]
        y_points_out_road = y_points[out_road]
        x_out_road_img, y_out_road_img = self.pcl2xy_plane(x_points_out_road,
                                                           y_points_out_road)

        pcl_anno_bev_img[y_road_img, x_road_img] = 255
        pcl_anno_bev_img[y_out_road_img, x_out_road_img] = 100
        return pcl_anno_bev_img

    @staticmethod
    def add_color_pcd(path1, path2, point_road_index):
        """
        Helper function to use different color to distinguish points on free 
        road or out free road.
        :param path1: where pointcloud is stored in .pcd format.
        :param path2: storage path after function add_color_pcd.
        :param point_road_index: index of points on free road in pointcloud 
                                 matrix.
        """
        with open(path1, 'rb') as fopen:
            lines = fopen.readlines()
        lines[2] = lines[2].split('\n')[0] + ' rgb\n'
        lines[3] = lines[3].split('\n')[0] + ' 4\n'
        lines[4] = lines[4].split('\n')[0] + ' I\n'
        lines[5] = lines[5].split('\n')[0] + ' 1\n'
        for i in range(len(lines)-11):
            if i in point_road_index:
                # 0xFFFF00: yellow, free road surface;
                # 0xFF0019: red, non-free road surface.
                lines[i + 11] = lines[i + 11].split('\n')[0] + ' ' + str(
                    0xFFFF00) + '\n'
            else:
                lines[i + 11] = lines[i + 11].split('\n')[0] + ' ' + str(
                    0xFF0019) + '\n'
        with open(path2, 'wb') as fwrite:
            fwrite.writelines(lines)

    @staticmethod
    def img_overlay_display(img1, img2):
        """
        Helper function to overlay two images to display.
        :param img1, img2: Images you want to overlay to display.
        :return: added image.
        """
        # img_1 has 3 channels, img2 has 1. 
        img_2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), 
                           interpolation=cv2.INTER_AREA)
        img_1 = np.array([[int(sum(img1[i][j])/3) for j in range(len(img1[i]))]
                          for i in range(len(img1))], dtype=np.uint8)
        alpha = 0.3
        beta = 1 - alpha
        gamma = 0
        img_add = cv2.addWeighted(img_1, alpha, img_2, beta, gamma)
        return img_add

    def get_line(self, original_point, box, x_0, y_0, pixel_x):
        """
        Assume that the obstacles and the areas behind the obstacles are all
        non-free road.
        Get a fit line from the lidar emission point to the obstacle boundary.       
        :param original_point: position of lidar emission point.
        :param box: representation of obstacle.
        :param x_0: boundary of box.
        :param y_0: boundary of box
        :param pixel_x: x range of area behind the obstacle.
        :return: a fit line, that is y range of area behind the obstacle.
        """
        y_line = int(original_point[1] -
                     (original_point[0] - pixel_x)*
                     (original_point[1] - box[1][y_0]) /
                     (original_point[0] - box[0][x_0])) + 1
        return y_line

    def get_bev_image(self, velodyne_path):
        if not os.path.exists(velodyne_path):
            raise ValueError(velodyne_path, "not Found")
        filtered_points = self.read_bin(velodyne_path)
        bev_image = self.pcl_2_bev(filtered_points)
        bev_image = cv2.merge([bev_image, bev_image, bev_image])
        return bev_image

###################  affine trainsform  ###################
def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        return trans, trans_inv
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])
