import json
import numpy as np
import cv2


def read_json(kp_labels):
    kp_positions = np.zeros((21, 2), dtype='float32')
    # read ezxr_render json label
    ps = [kp_labels['f52'][0]['x'] + 0.5, kp_labels['f52'][0]['y'] + 0.5]
    kp_positions[0] = np.asarray(ps)
    jid = 1
    for fid in range(5):
        ps = kp_labels['f%d2' % fid]
        for i in range(4):
            p = [ps[i]['x'] + .5, ps[i]['y'] + 0.5]
            kp_positions[jid] = np.asarray(p)
            jid += 1
    return kp_positions


def generate_jcd(kp_positions, handsize_normal=False):
    """
    generate_jcd is a function that generate jcd

    Args:
        :param kp_positions: keypoints position in 2d (21,2)
        :param handsize_normal: whether use handsize(kp0-kp9) to normalize data, default using (/100)
    Returns:
        jcd representation of keypoints (210,)
    Raises:
        IOError: An error occurred accessing the bigtable.Table object.
    """
    dist_tri = np.zeros(210)
    for i in range(1, 21):
        for j in range(i):
            dist_tri[(i * (i - 1)) // 2 + j] = np.linalg.norm(kp_positions[i, :] - kp_positions[j, :])
    if handsize_normal:
        dist_norm = np.linalg.norm(kp_positions[9, :] - kp_positions[0, :])
        dist_tri = dist_tri / dist_norm
    else:
        dist_tri = dist_tri / 100
    return dist_tri


def generate_heatmap(joints, kernel_size, sigma):
    """
    generate_heatmap is a function that generate heatmap

    Args:

        :param joints: keypoints position in 2d(21*2)
        :param kernel_size: kernel size of the heatmap
        :param sigma: std
    Returns:
        heatmap representation of keypoints (H, W, num_kps)
    Raises:
        IOError: An error occurred accessing the bigtable.Table object.
    """
    target = np.zeros((21, 250, 250), dtype=np.float32)

    tmp_size = kernel_size
    for joint_id in range(21):
        mu_x = int(joints[joint_id][0] + 0.5)
        mu_y = int(joints[joint_id][1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], 250) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], 250) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], 250)
        img_y = max(0, ul[1]), min(br[1], 250)

        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target.transpose((1,2,0))


if __name__ == '__main__':
    kernel_size = 20
    sigma = 8.0
    # joints = np.full((21, 2), 125)
    with open('D:\Hand-data/HUMBI/data/Hand_381_453_updat/subject_391/hand/00000025/left/image0000075.json', 'r') as file:
        joints_json = json.load(file)
    joints = read_json(joints_json)
    heatmap = generate_heatmap(joints, kernel_size, sigma)
    print(heatmap[1])
    # heatmap = np.sum(heatmap, axis=0)
    cv2.imshow('1',heatmap)
    cv2.waitKey(0)
