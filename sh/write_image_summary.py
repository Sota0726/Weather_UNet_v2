import argparse
import os
import sys

import numpy as np
import cv2
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/results/c_UNet/tensorboard')
parser.add_argument('--log_path', type=str,
                    default='runs/Apr11_20-10-29_DLBOX2_lr-0.0001_bs-16_ne-50_name-cUNet_w-e-res101-0408_train-D1T1_adam-b1-09_wloss-mse_train200k-test500'
                    )
args = parser.parse_args()

sys.path.append(os.getcwd())

output_path = os.path.join(args.output_dir, args.log_path.split('/')[-1])
os.makedirs(output_path, exist_ok=True)

path = args.log_path  # Tensorboard ログファイル
event_acc = EventAccumulator(path, size_guidance={'images': 0})
event_acc.Reload()  # ログファイルのサイズによっては非常に時間がかかる

for tag in event_acc.Tags()['images']:
    events = event_acc.Images(tag)
    tag_name = tag.replace('/', '_')
    for index, event in enumerate(events):
        # 画像はエンコードされているので戻す
        s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
        image = cv2.imdecode(s, cv2.IMREAD_COLOR)  # カラー画像の場合
        # 保存
        output_path_ = os.path.join(output_path, '{}_{:04}.jpg'.format(tag_name, index))
        cv2.imwrite(output_path_, image)
