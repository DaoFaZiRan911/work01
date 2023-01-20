import os
import cv2
import numpy as np
import math
from collections import defaultdict
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow
import face_recognition
"""
该文件集成了对面部图像进行人脸对齐的一系列操作，基本思路是利用开源项目“face_recognition”
（可以在安装好dlib、cmake后直接pip3 install face_recognition）对图片中的人脸进行识别，进一步匹配关键点（landmarks），
对图像进行旋转，将眼睛连线置换到水平位置，并记录旋转的角度，再将关键点整体旋转相同的角度，与对齐后的图像重新标定。
注意：数据集中可能会存在无法标定关键点的图像，比如眼部遮挡、不露眼睛的侧脸图、背影图等等。在本程序中我将此类特殊情况做了忽略处理，
即跳过这些无法对齐或难以对齐的数据。换言之，处理后的数据集长度将小于等于原数据集。
"""


# 人脸对齐函数
def align_face(image_array, landmarks):
    """
    align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = (float((left_eye_center[0] + right_eye_center[0]) // 2),
                  float((left_eye_center[1] + right_eye_center[1]) // 2))
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle


# 人脸关键点可视化
def visualize_landmark(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    for facial_feature in landmarks.keys():
        draw.point(landmarks[facial_feature])
    imshow(origin_img)


# 关键点旋转
def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)


# 旋转标记点
def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            rotated_landmark = rotate(origin=eye_center, point=landmark, angle=angle, row=row)
            rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks


# 人脸裁剪
def corp_face(image_array, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    """

    eye_landmark = np.concatenate([np.array(landmarks['left_eye']),
                                   np.array(landmarks['right_eye'])])
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    lip_landmark = np.concatenate([np.array(landmarks['top_lip']),
                                   np.array(landmarks['bottom_lip'])])
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 30 / 35
    bottom = lip_center[1] + mid_part

    w = h = bottom - top
    x_min = np.min(landmarks['chin'], axis=0)[0]
    x_max = np.max(landmarks['chin'], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top


# 关键点旋转变换
def transfer_landmark(landmarks, left, top):
    """transfer landmarks to fit the cropped face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param left: left coordinates of cropping
    :param top: top coordinates of cropping
    :return: transferred_landmarks with the same structure with landmarks, but different values
    """
    transferred_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            transferred_landmark = (landmark[0] - left, landmark[1] - top)
            transferred_landmarks[facial_feature].append(transferred_landmark)
    return transferred_landmarks


def align_imgs(txt):
    fh = open(txt, 'r')  # 读取文件
    imgs = []  # 用来存储路径与标签
    # 一行一行的读取
    for line in fh:
        line = line.rstrip()  # 这一行就是图像的路径，以及标签
        words = line.split(' ')
        imgs.append(
            (os.path.join(r'D:\Python\AVEC2014\xiamen_img', words[0])))  # 路径添加到列表中
    for line in imgs:
        image_array = cv2.imread(line)
        b, g, r = cv2.split(image_array)  # 分别提取B、G、R通道
        image_array = cv2.merge([r, g, b])  # 重新组合为R、G、B
        face_landmarks_list = face_recognition.face_landmarks(image_array, model="large")
        if len(face_landmarks_list) == 0:
            continue
        face_landmarks_dict = face_landmarks_list[0]
        aligned_face, eye_center, angle = align_face(image_array=image_array, landmarks=face_landmarks_dict)
        rotated_landmarks = rotate_landmarks(landmarks=face_landmarks_dict,
                                             eye_center=eye_center, angle=angle, row=image_array.shape[0])
        cropped_face, left, top = corp_face(image_array=aligned_face, landmarks=rotated_landmarks)
        transferred_landmarks = transfer_landmark(landmarks=rotated_landmarks, left=left, top=top)
        origin_img = Image.fromarray(cropped_face)
        draw = ImageDraw.Draw(origin_img)
        # for facial_feature in transferred_landmarks.keys():
        #     draw.point(transferred_landmarks[facial_feature])
        the_folder = line.split('\\')[-2]
        the_pic_name = line.split('\\')[-1]
        save_root_address = r'D:\Python\AVEC2014\ALIGN_xiamen'
        root_ads = os.path.join(save_root_address, the_folder)
        # if the_pic_name == '001.jpg':
        #     os.mkdir(root_ads)
        # else:
        #     print('*********')
        save_path = os.path.join(root_ads, the_pic_name)
        origin_img.save(save_path)
        print('This aligned image has been saved->', save_path)


if __name__ == '__main__':
    txt_path = r'D:\Python\AVEC2014\xiamenImg.txt'
    align_imgs(txt_path)


