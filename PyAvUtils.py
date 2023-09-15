# -*- coding: utf-8 -*-

"""
@author: tjm
@software: PyCharm
@file: PyAvUtils.py
@time: 2022/4/15 14:09
"""

import av
import os
import shutil


# 视频提取关键帧工具类（支持批量视频）
class PyAvUtils:
    def __init__(self, video_dir, keyframe_dir):
        self.video_dir = video_dir
        self.keyframe_dir = keyframe_dir

    # 提取关键帧并保存
    def do_video2image(self):
        for video_name in os.listdir(self.video_dir):
            print(video_name)
            video_keyframe_dir = os.path.join(self.keyframe_dir, video_name)
            if not os.path.exists(video_keyframe_dir):
                os.makedirs(video_keyframe_dir)
                print("已自动创建：", video_keyframe_dir)
            else:
                print("目录已存在")
            # 提取关键帧
            self.extract_video(video_name)

    # 定义提取视频关键帧的函数
    def extract_video(self, filename):
        cur_video_path = os.path.join(self.video_dir, filename)
        cur_video_keyframe_dir = os.path.join(self.keyframe_dir, filename)
        container = av.open(cur_video_path)
        # Signal that we only want to look at keyframes.
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = 'NONKEY'
        for frame in container.decode(stream):
            frame.to_image().save(cur_video_keyframe_dir + '/' + 'frame.{:04d}.jpg'.format(frame.pts),
                                  quality=90)


if __name__ == '__main__':
    pa = PyAvUtils('./videos', './images')
    pa.do_video2image()