import argparse
import os
import sys

import numpy as np
import cv2
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/results/c_UNet/tensorboard')
parser.add_argument('--log_path', type=str,
                    default='runs/Nov24_13-08-43_647e57fb8e44_lr-2*0.0001_bs-32_ne-150_name-Flickr_cUNet_w-c_res101-1122e15_Wosampler-BS32_sampler-False_loss_lamda-c1-w1-CE_b1-0.5_b2-0.9_GDratio-8_amp-True_MGpu-True'
                    )
args = parser.parse_args()

sys.path.append(os.getcwd())
output_path = os.path.join(args.output_dir, args.log_path.split('/')[-1])
print(output_path)
input()
os.makedirs(output_path, exist_ok=True)

path = args.log_path  # Tensorboard ログファイル
event_acc = EventAccumulator(path, size_guidance={'images': 0})
event_acc.Reload()  # ログファイルのサイズによっては非常に時間がかかる

for tag in event_acc.Tags()['images']:
    events = event_acc.Images(tag)
    tag_name = tag.replace('/', '_')
    for index, event in tqdm(enumerate(events), total=len(events)):
        # 画像はエンコードされているので戻す
        s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
        image = cv2.imdecode(s, cv2.IMREAD_COLOR)  # カラー画像の場合
        # 保存
        output_path_ = os.path.join(output_path, '{}_{:04}.jpg'.format(tag_name, index))
        cv2.imwrite(output_path_, image)
