# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '19-11-22'
__email__ = 'kohou.wang@cloudminds.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
inference the gender using trained models.
"""

import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from models.Resnext import resnext101_32x16d_swsl, resnext101_32x4d_swsl, \
    resnext50_32x4d_swsl
import argparse
import time
import json
import torch
from torchvision import transforms as T
from PIL import Image
from datasets.read_data import ReadImageDataset
from torch.utils.data import DataLoader
from models.EfficientNet import EfficientNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_ = T.Compose([
    # T.Resize((224 + 32, 224 + 32)),
    # T.CenterCrop(224),
    T.Resize((224, 224)),
    # T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def inference_single_image(model_, image_path_, input_size_=64, transform__=transform_, cpu=False):
    # import ipdb
    # ipdb.set_trace()
    image_ = Image.open(image_path_).convert('RGB')
    # image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    
    image_ = transform__(image_)
    # image_ = torch.tensor(np.asarray(image_), dtype=torch.float32)
    # print("image before:{}".format(image_))
    # print("image:{}".format((image_ - 255 * 0.485) / (255 * 0.229)))
    
    # image_9 = torch.tensor(np.asarray(image_9))
    # image_8 = torch.tensor(np.asarray(image_8))
    
    image_ = torch.unsqueeze(image_, 0)
    
    # import ipdb
    # ipdb.set_trace()
    start_time_ = time.time()
    if not cpu:
        image_ = image_.cuda()
    # print("Image_:{}".format(image_))
    results_ = model_(image_)
    return results_, time.time() - start_time_


def inference_single_image_with_models(models_, image_path_, input_size_=64, transform__=transform_, cpu=False):
    image_ = Image.open(image_path_).convert('RGB')
    # image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    
    image_ = transform__(image_)
    
    image_ = torch.unsqueeze(image_, 0)
    start_time_ = time.time()
    results_ = torch.tensor([[0., 0.]])
    
    if not cpu:
        image_ = image_.cuda()
        results_ = results_.cuda()
    # print("Image_:{}".format(image_))
    # import ipdb
    # ipdb.set_trace()
    for model_ in models_:
        results_ = results_ + model_(image_)
    return results_, time.time() - start_time_


if __name__ == "__main__":
    image_file_path = "/home/kohou/cvgames/interest/contest/MARS/FuBuFu/data/data/test"
    model_file = "/home/kohou/cvgames/interest/contest/MARS/FuBuFu/trained_models/fcanet/best_epoch_FinetuneOnmixup_input224_DifferentLR__False_batch8_lr0.002_epoch30.pth"
    inferenced_result = '/home/kohou/cvgames/interest/contest/MARS/FuBuFu/submissions/fcanet101/result_300Nocrop_lr0.002.csv'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed, dir or a single image.")
    parser.add_argument("--graph", help="graph/model to be executed")
    
    args = parser.parse_args()
    
    if args.graph:
        model_file = args.graph
    if args.image:
        image_file_path = args.image
    
    # input_size = 224
    # input_size = 400
    input_size = 300  # efficientbetb3
    # input_size = 456  # efficientnet_b5
    batch_size = 1
    num_classes = 2
    use_cpu = False
    use_multi_model = False
    use_flipped_image = True
    alpha = 0.5
    
    # inference_model = EfficientNet.from_name('efficientnet-b3', override_params={'num_classes': num_classes})
    # inference_model =  resnext101_32x16d_swsl(num_classes=num_classes)
    inference_model = resnext101_32x4d_swsl(num_classes=num_classes)
    # inference_model = resnext50_32x4d_swsl(num_classes=num_classes)
    if use_cpu:
        loaded_model = torch.load(model_file,
                                  map_location=torch.device('cpu')
                                  )
    else:
        loaded_model = torch.load(model_file,
                                  # map_location=torch.device('cpu')
                                  )
    inference_model.load_state_dict(loaded_model)
    if not use_cpu:
        inference_model = inference_model.to(device)
    inference_model.eval()
    
    start_time = time.time()
    
    del loaded_model
    # batch inference.
    inference_data_gen = ReadImageDataset(os.listdir(image_file_path), image_file_path, input_size=input_size,
                                          mode='test', flipped_image=use_flipped_image)
    inference_data_loader = DataLoader(inference_data_gen, batch_size=batch_size,
                                       shuffle=False, pin_memory=True, num_workers=2)
    
    single_result_dict = {"image_name": '',
                          "category": '',
                          "score": 0.}
    single_result_dict_list = []
    results_list = []
    dst_file = open(inferenced_result, 'w')
    if use_flipped_image:
        print('use flipped images...')
        for index_, (image_flipped_, images_, image_names_) in enumerate(inference_data_loader):
            if not use_cpu:
                image_flipped_ = image_flipped_.to(device)
                images_ = images_.to(device)
            outputs = inference_model(images_)
            outputs = outputs + inference_model(image_flipped_)
            
            _, preds = torch.max(outputs, 1)
            results_list.append(preds.tolist()[0])
            for temp_index in range(len(image_names_)):
                # import ipdb
                # ipdb.set_trace()
                single_result_dict['image_name'] = image_names_[temp_index]
                single_result_dict['category'] = preds.tolist()[temp_index]
                single_result_dict['score'] = float(torch.nn.functional.softmax(outputs)[temp_index][preds[temp_index]])
                
                single_result_dict_list.append(single_result_dict.copy())
            if len(results_list) % 100 == 0:
                print("{} predicted...".format(len(results_list)))
    
    # json.dump(single_result_dict_list, dst_file)
    # dst_file.write("]")
    
    with open('/home/kohou/cvgames/interest/contest/MARS/FuBuFu/data/data/result.csv', 'r') as submission_template:
        while True:
            image_name = submission_template.readline().strip()
            if len(image_name) == 0:
                break
            for single_result_dict in single_result_dict_list:
                if image_name == single_result_dict['image_name']:
                    dst_file.write("{},{}\n".format(image_name, single_result_dict['category']))
    
    dst_file.close()
    end_time = time.time()
    print("cost {} s".format(end_time - start_time))
