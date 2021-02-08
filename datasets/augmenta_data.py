# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '19-12-18'
__email__ = 'oukohou@outlook.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
augment datasets.
"""
from imgaug import augmenters as iaa
import imgaug as ia
from PIL import Image
import os
import cv2
import json

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
augment_img = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.2),
    sometimes(iaa.CropAndPad(
        percent=(0., 0.1),
        pad_mode=ia.ALL,
        pad_cval=(0, 255)
    )),
    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
        rotate=(-25, 25),  # rotate by -45 to +45 degrees
        shear=(-16, 16),  # shear by -16 to +16 degrees
        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
        mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    )),
    iaa.SomeOf((0, 7), [
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270),
        iaa.Affine(shear=(-16, 16)),
        iaa.PerspectiveTransform(0.025),
        iaa.GammaContrast(),
        iaa.MultiplyHueAndSaturation(mul=1.25),
    
    ]),
    # iaa.OneOf([
    #     iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
    #     iaa.AverageBlur(k=(2, 7)),  # blur image using local means with kernel sizes between 2 and 7
    #     iaa.MedianBlur(k=(3, 11)),  # blur image using local medians with kernel sizes between 2 and 7
    # ]),
    # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
], random_order=True)


def augmenta_and_save(src_dir_, dst_dir_, times_to_aug_, json_label_):
    josn_dict_ = json.load(open(json_label_))
    for value_ in josn_dict_.values():
        if not os.path.exists(os.path.join(dst_dir_, value_)):
            os.makedirs(os.path.join(dst_dir_, value_))
    
    count_ = 0
    for subpath_, dirnames_, filenames_ in os.walk(src_dir_):
        for filename_ in filenames_:
            for index_ in range(times_to_aug_):
                img_aug = augment_img.augment_image(cv2.imread((os.path.join(subpath_, filename_))))
                cv2.imwrite(os.path.join(dst_dir_, subpath_.split(os.sep)[-2], subpath_.split(os.sep)[-1],
                                         "{}_{}".format(index_, filename_)),
                            img_aug)
            count_ = count_ + 1
            if count_ % 1000 == 0:
                print("processed {}...".format(count_))
    print("processed {}...".format(count_))


if __name__ == "__main__":
    augmenta_and_save('/home/data/CVAR-B/study/interest/contest/HUAWEI/foods/data/merged_data/merged_original_baidu_google_v2_150_cleaned',
                      '/home/data/CVAR-B/study/interest/contest/HUAWEI/foods/data/merged_data/augmented_merged_2times',
                      2,
                      '/home/CVAR-B/study/interest/contest/HUAWEI/foods/data/train_data/label_id_name.json')
