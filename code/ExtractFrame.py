import os
import cv2

# parameters
videos_path = r'D:\Python\AVEC2014\BDI-II\4'  # 视频文件夹地址
saved_path = r'D:\Python\AVEC2014\xiamen_img\4'  # 保存的图像文件夹地址


def save2img(video_path, img_path):
    frame_ind = 1  # 帧索引
    frame_gap = 10  # 保存视频的帧间隔
    exist_ok = True
    obj = cv2.VideoCapture(video_path)
    while(exist_ok):
        exist_ok, frame = obj.read()
        print(f'the frame index of video {video_path} is:', frame_ind)
        if exist_ok == True and frame_ind % frame_gap == 0:
            cv2.imwrite(f'{img_path}/{frame_ind}.jpg', frame)
        frame_ind = frame_ind + 1


for video_name in os.listdir(videos_path):
    if os.path.exists(os.path.join(saved_path, video_name.split('.')[0])):
        print(f"You need to delete the filedoc: {os.path.join(saved_path, video_name.split('.')[0])}")
        break
    os.makedirs(os.path.join(saved_path, video_name.split('.')[0]))
    img_path = os.path.join(saved_path, video_name.split('.')[0])
    video_path = os.path.join(videos_path, video_name)
    save2img(video_path, img_path)

