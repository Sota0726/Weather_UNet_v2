import pandas as pd
import os
import numpy as np
import cv2
from tqdm import tqdm
# from multiprocessing import Pool


def main(photo):
    img_path = os.path.join(img_root, str(photo)+'.jpg')
    img = cv2.imread(img_path)
    if (img[:,:,0] == img[:,:,0]).all() and (img[:,:,1] == img[:,:,2]).all():
        d_ind = df[df.photo == photo].index
        df.drop(index=d_ind, inplace=True)


if __name__=="__main__":
    img_root = '/mnt/fs2/2019/Takamuro/db/photos_usa_2016'
    df = pd.read_pickle('/mnt/fs2/2019/Takamuro/m2_research/flicker_data/from_nitta/param03/for_transfer-train_train-low-consis-214938_test-high-consis-500.pkl')
    cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed']
    c_cols = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist']

    photo_list = df['photo'].dropna().to_list()
    # pool = Pool(processes=6)
    # for _ in tqdm(pool.imap_unordered(main, photo_list)):
    #     None
    [main(_) for _ in tqdm(photo_list)]
    
    df.to_pickle('/mnt/fs2/2019/Takamuro/m2_research/flicker_data/from_nitta/param03/temp_WoGray_for_transfer-esttrain214938_test500.pkl')
    print(df)
