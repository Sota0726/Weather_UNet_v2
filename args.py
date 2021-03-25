import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_root',
        type=str,
        default='/mnt/HDD8T/takamuro/dataset/photos_usa_224_2016-2017'
    )
    parser.add_argument(
        '--i2w_root',
        type=str,
        default='/mnt/fs2/2018/matsuzaki/dataset_fromnitta/Image/'
    )
    parser.add_argument(
        '--vid_root',
        type=str,
        default='/mnt/HDD8T/takamuro/dataset/timelapse_frames'
    )
    parser.add_argument(
        '--celebA_root',
        type=str,
        default='/mnt/fs2/2019/Takamuro/db/CelebA/Img/img_align_celeba'
    )
    parser.add_argument(
        '--csv_root',
        type=str,
        default='/mnt/HDD8T/takamuro/dataset/wwo/2016_2017/'
    )
    parser.add_argument(
        '--pkl_path',
        type=str,
        default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/lambda_0/'
        'outdoor_all_dbdate_wwo_weather_2016_17_delnoise_WoPerson_sky-10_L-05.pkl'
    )
    parser.add_argument(
        '--test_pkl_path',
        type=str,
        default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/lambda_0/'
        'for_test_training.pkl'
    )
    parser.add_argument(
        '--i2w_pkl_path',
        type=str,
        default='/mnt/fs2/2019/Takamuro/db/i2w/sepalated_data.pkl'
    )
    parser.add_argument(
        '--celebA_pkl_path',
        type=str,
        default='/mnt/fs2/2019/Takamuro/db/CelebA/Anno/list_attr_celeba_add-mode.pkl'
    )
    parser.add_argument(
        '--estimator_path',
        type=str,
        default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/estimator/'
        'est_res101-1203_sampler_pre_WoPerson_sky-10_L-05/est_resnet101_15_step62240.pt'
    )
    parser.add_argument(
        '--classifier_path',
        type=str,
        default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/classifier/'
        'cls_res101_1122_NotPreTrain/resnet101_epoch15_step59312.pt'
    )
    parser.add_argument('--save_dir', type=str, default='cp/transfer')
    parser.add_argument('--est_save_dir', type=str, default='cp/estimator')
    parser.add_argument('--cls_save_dir', type=str, default='cp/classifier')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['flickr', 'i2w', 'celebA'],
        default='flickr')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--data_mode', type=str, choices=['T', 'E'], default='T')
    parser.add_argument('--name', type=str, default='cUNet')
    # Nmaing rule : cUNet_[c(classifier) or e(estimator)]_[detail of condition]_[epoch]_[step]
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('-b1', '--adam_beta1', type=float, default=0.5)
    parser.add_argument('-b2', '--adam_beta2', type=float, default=0.9)
    parser.add_argument('-e', '--epsilon', type=float, default=1e-7)
    parser.add_argument('--lmda', type=float, default=None)
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', '-bs', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--GD_train_ratio', type=int, default=5)
    parser.add_argument('--train_data_ratio', type=float, default=0.5)
    parser.add_argument(
        '--sampler',
        choices=['time', 'class', 'none'],
        default='none'
    )
    parser.add_argument('--resume_cp', type=str)
    parser.add_argument('-ww', '--wloss_weight', type=int, nargs=8, default=[1, 1, 1, 1, 1, 1, 1, 1])
    parser.add_argument(
        '-wt', '--wloss_type',
        choices=['mse', 'CE', 'weightedMSE', 'L1', 'BCE'],
        default='mse'
    )
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--pre_trained', action='store_true')
    parser.add_argument(
        '-g', '--generator',
        type=str,
        choices=['cUNet', 'cUNetV2'],
        default='cUNet'
    )
    parser.add_argument(
        '-d', '--disc',
        type=str,
        choices=['SNDisc', 'SNDiscV2', 'SNRes64', 'SNRes'],
        default='SNDisc'
    )
    parser.add_argument(
        '--seq_disc',
        type=str,
        choices=['res10_3d', 'res18_3d', 'res34_3d'],
        default='res10_3d'
    )
    parser.add_argument(
        '--predictor',
        type=str,
        choices=['resnet', 'mobilenet', 'resnext', 'wideresnet'],
        default='resnet'
    )
    args = parser.parse_args()
    # args = parser.parse_args(args=['--name', 'debug', '--multi_gpu', '--gpu', '0,1,2,3'])

    return args
