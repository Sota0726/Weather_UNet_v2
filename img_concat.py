from glob import glob
import cv2
import os

root_path = '/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/results/eval_transfer/seq/Seq_0114_cUNet_w-e_res101-1203e15_timesamp-NightShift_D-SNRes64_sampler-time_GDratio1-5_adam-b10.5-b20.9_ep1e-07_lr-0.0001_bs-8_ne-150/cUNet_est_e0032_s1030000/osaka_mino_vid/'


t_imgs = glob(os.path.join(root_path, 'transform', '*.jpg'))
o_imgs = glob(os.path.join(root_path, 'original', '*.jpg'))

t_l = []
o_l = []
for i, img in enumerate(zip(t_imgs[::2], o_imgs[::2]), 1):
    t_img = img[0]
    o_img = img[1]
    o = cv2.imread(o_img)
    t = cv2.imread(t_img)
    t_l.append(t)
    o_l.append(o)

    if i % 13 == 0:
        o_ = cv2.hconcat(o_l)
        t_ = cv2.hconcat(t_l)
        cv2.imwrite(os.path.join(root_path, 'o_{}.jpg'.format(i)), o_)
        cv2.imwrite(os.path.join(root_path, 't_{}.jpg'.format(i)), t_)
        t_l = []
        o_l = []