# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '19-11-22'
__email__ = 'oukohou@outlook.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
compare the difference between two kinds of reading images.
- read using PIL's Image
- read using opencv and then convert to RGB order by pytorch's T.ToPILImage()


conclusion: it's much different! most time opencv can work consistently on different platforms.
"""

import cv2
from PIL import Image
from torchvision import transforms as T
import os
from matplotlib import pyplot as plt


def compare_(image_path_):
    image_ = cv2.imread(image_path_)
    image_ = T.Compose([
        T.ToPILImage(),
        # T.Resize((64, 64)),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(image_)
    PIL_image = Image.open(image_path_)
    PIL_image = T.Compose([
        # T.Resize((64, 64)),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(PIL_image)
    import ipdb
    ipdb.set_trace()


def see_width_height_distribution(src_dir_):
    width_list = []
    hegith_list = []
    
    count_ = 0
    label_file = open('/home/kohou/cvgames/interest/contest/MARS/FuBuFu/data/data/train_label.csv','r')
    labels = label_file.readlines()
    label_list = []
    ratio_list = []
    for image_name_ in os.listdir(src_dir_):
        image_ = cv2.imread(os.path.join(src_dir_, image_name_))
        height_, width_, _ = image_.shape
        
        for temp_label in labels:
            if image_name_ in temp_label:
                label_list.append(int(temp_label.strip().split(',')[-1].split('.')[0]))
                break
        ratio_list.append(height_/width_)
        width_list.append(width_)
        hegith_list.append(height_)
        count_ = count_ + 1
        if count_ % 1000 == 0:
            print("{} images processed...".format(count_))
    print("max width:{}, max heigth:{}".format(max(width_list), max(hegith_list)))
    # plt.plot(width_list, hegith_list)
    # plt.scatter(width_list, hegith_list, )
    plt.scatter( label_list, ratio_list)
    # plt.xlim(-0.5, 650)
    # plt.ylim(-0.5, 650)
    plt.show()


if __name__ == "__main__":
    # compare_(
    #     "/home/data/CVAR-B/study/projects/face_properties/age_estimation/datasets/megaage_asion/megaage_asian/megaage_asian/cropped_images/test/0.8_1.jpg")
    see_width_height_distribution('/home/kohou/cvgames/interest/contest/MARS/FuBuFu/data/data/train')
