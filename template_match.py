# pts1 = np.float32([[946, 1190], [2770, 1203], [1149, 457]])
# pts2 = np.float32([[1003, 1142], [2800, 1197], [1202, 369]])
# pts1 = np.float32([[int(1213.885009765625), int(974.0836791992188)], [int(1341.3397216796875), int(2788.175048828125)], [int(807.9646606445312), int(1159.915283203125)]])
# pts2 = np.float32([[int(1142 + 37.72761917114258), int(1003 + 28.4157657623291)], [int(1197 + 138.05038452148438), int(2800 + 17.299684524536133)], [int(369 + 350.0006408691406), int(1202 + 10.236589431762695)]])
import os

import numpy as np
import cv2
from PIL import Image
import math


def root_sift(kp, des, eps=1e-7):
    if len(kp) == 0:
        return [], None

        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
    des /= (des.sum(axis=1, keepdims=True) + eps)
    des = np.sqrt(des)
    # des /= (np.linalg.norm(des, axis=1, ord=2) + eps)

    # return a tuple of the keypoints and descriptors
    return kp, des


def match_template(template_img_name, ori_img):
    # sift = cv2.xfeatures2d.SURF_create()
    sift = cv2.xfeatures2d.SIFT_create()
    # fast = cv2.FastFeatureDetector_create()
    # sift = cv2.ORB_create()
    # template_img = cv2.imdecode(np.fromfile(template_img_name, dtype=np.uint8), -1)
    # gray1 = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)  # 灰度处理图像
    template_img = Image.open(template_img_name)
    gray1 = np.array(template_img.convert('L'))
    kp1, des1 = sift.detectAndCompute(gray1, None)  # des是描述子
    kp1, des1 = root_sift(kp1, des1)

    # ori_img = cv2.imdecode(np.fromfile(ori_img_name, dtype=np.uint8), -1)
    # gray2 = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)  # 灰度处理图像
    # ori_img = Image.open(ori_img_name)
    gray2 = np.array(ori_img.convert('L'))
    kp2, des2 = sift.detectAndCompute(gray2, None)  # des是描述子
    kp2, des2 = root_sift(kp2, des2)

    # flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
    # matches = flann.knnMatch(des1, des2, k=2)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    print(111111111111, matches[0].distance, matches[0].distance)
    numGoodMatches = math.ceil(len(matches) * 0.15)
    matches = matches[:numGoodMatches]
    # 调整ratio
    # good = [m1 for (m1, m2) in matches if m1.distance < 0.5 * m2.distance]
    good = matches
    # good = [m1 for m1 in matches if m1.distance <= matches[0].distance * 2]
    ori_position = template_img_name.split('/')[-1].split('.')[0].split('_')
    if len(good):
        # src_pts = np.float32(
        #     [[kp1[m.queryIdx].pt[0] + int(ori_position[0]), kp1[m.queryIdx].pt[1] + int(ori_position[1])] for m in
        #      good])
        # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # matchesMask = mask.ravel().tolist()
        # for i, j in zip(matchesMask, good):
        #     if i == 0:
        #         good.remove(j)
        # good = sorted(good, key=lambda i: i.distance)

        src_pts = np.float32(
            [[kp1[m.queryIdx].pt[0] + int(ori_position[0]), kp1[m.queryIdx].pt[1] + int(ori_position[1])] for m in
             good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
        print(len(good))
        return src_pts, dst_pts

    else:
        print("Not enough matches are found")
        matchesMask = None


def affine_transform(img, template_img_name, pts1, pts2):
    # img = cv2.imdecode(np.fromfile(ori_img_name, dtype=np.uint8), -1)
    img = np.array(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    template_img = cv2.imdecode(np.fromfile(template_img_name, dtype=np.uint8), -1)
    # M = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    h, w = template_img.shape[:2]
    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    M = cv2.getPerspectiveTransform(np.float32(dst), pts)
    # M = cv2.getPerspectiveTransform(pts, np.float32(dst))

    found = cv2.warpPerspective(img, M, (w, h))
    # M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    # dst = cv2.warpPerspective(img, M, (template_img.shape[1], template_img.shape[0]))
    return found


from ctpn.ctpn_blstm_test import text_predict


def match_template_calibration_test1(template_img_name, ori_img, scale):
    # sift = cv2.xfeatures2d.SURF_create()
    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.ORB_create()
    # sift = cv2.xfeatures2d.AKAZE_create()

    # template_img = cv2.imdecode(np.fromfile(template_img_name, dtype=np.uint8), -1)
    # gray1 = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)  # 灰度处理图像
    template_img = Image.open(template_img_name)
    gray1 = np.array(template_img.convert('L'))
    kp1, des1 = sift.detectAndCompute(gray1, None)  # des是描述子
    kp1, des1 = root_sift(kp1, des1)

    # ori_img = cv2.imdecode(np.fromfile(ori_img, dtype=np.uint8), -1)
    # gray2 = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)  # 灰度处理图像
    gray2 = np.array(Image.fromarray(ori_img).convert('L'))
    kp2, des2 = sift.detectAndCompute(gray2, None)  # des是描述子
    kp2, des2 = root_sift(kp2, des2)

    # flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
    # matches = flann.knnMatch(des1, des2, k=2)
    # matches = sorted(matches, key=lambda x: x[0].distance)
    print(type(des1), type(des2))
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # numGoodMatches = math.ceil(len(matches) * 0.15)
    # matches = matches[:numGoodMatches]

    # good = [m1 for m1 in matches if m1.distance < matches[0].distance * 2]
    ori_position = template_img_name.split('/')[-1].split('.')[0].split('_')
    img_shape = gray1.shape
    good = matches
    if len(good):
        # src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        src_pts = np.float32(
            [[(int(ori_position[0]) + kp1[m.queryIdx].pt[0]), kp1[m.queryIdx].pt[1] + int(ori_position[1])] for m in
             good])
        dst_pts = np.float32([[int(ori_position[0]) - img_shape[1] * scale + kp2[m.trainIdx].pt[0],
                               int(ori_position[1]) - img_shape[0] * scale + kp2[m.trainIdx].pt[1]] for m in good])
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        for i, j in zip(matchesMask, good):
            if i == 0:
                good.remove(j)
        good = sorted(good, key=lambda i: i.distance)
        # if good[1].distance - good[0].distance > 10:
        #     print(111111111)
        #     idx = good[0].queryIdx
        #     idx2 = good[0].trainIdx
        # else:
        #     print(22222222222)
        idx = good[0].queryIdx
        idx2 = good[0].trainIdx
        # print(4444444444444444, good[0].distance, good[1].distance)
        # print(6666666666666,
        #       int(ori_position[0]) - img_shape[1] * scale + kp2[idx2].pt[0] - (int(ori_position[0]) + kp1[idx].pt[0]),
        #       int(ori_position[1]) - img_shape[0] * scale + kp2[idx2].pt[1] - (int(ori_position[1]) + kp1[idx].pt[1]))
        return int(ori_position[0]) - img_shape[1] * scale + kp2[idx2].pt[0] - (
                int(ori_position[0]) + kp1[idx].pt[0]), int(ori_position[1]) - img_shape[0] * scale + kp2[idx2].pt[
                   1] - (int(ori_position[1]) + kp1[idx].pt[1])
    else:
        print("Not enough matches are found")
        matchesMask = None
