import os
import argparse
import pickle
import glob
import random
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=100, help='A number of test data')
parser.add_argument('--root_dir', type=str, default='/mnt/fs2/2018/matsuzaki/dataset_fromnitta/Image/')
parser.add_argument('--out_path', type=str, default='sepalated_data.pkl')
parser.add_argument('--small_rate', type=float, default=1.0, help='for small data learning, matsuzaki setting is 0.3')
args = parser.parse_args()

print('Start parsing... Depending on the size of the directory, it may take a long time.')
class_dirs = os.listdir(args.root_dir)
class_dirs.remove('z-other')

dataset_dict = defaultdict(list)
half = lambda x: (x[:len(x) // 2], x[len(x) // 2:])

for class_dir in class_dirs:
    print(class_dir)
    class_path = os.path.join(args.root_dir, class_dir)
    imgs = glob.glob(os.path.join(class_path, '*.jpg'))
    random.shuffle(imgs)
    imgs = imgs[:int(len(imgs) * args.small_rate)]

    data_tr, data_va = half(imgs[args.N:])
    dataset_dict['test'] += imgs[:args.N]
    dataset_dict['train'] += data_tr
    dataset_dict['val'] += data_va

print("The split was successful.\n train:val:test = {}:{}:{}".format(len(dataset_dict['train']),
    len(dataset_dict['val']),
    len(dataset_dict['test'])))

with open(args.out_path, "wb") as f:
    pickle.dump(dataset_dict, f)


### original code written by matsuzaki

# print('Start parsing... Depending on the size of the directory, it may take a long time.')
# path_li = glob.glob(os.path.join(args.root_dir,'**'), recursive=True)
# print("Found {} images. {} data will be split at 2:1:1".format(len(path_li), args.N))
# random.shuffle(path_li)

# dict = defaultdict(list)
# for path in path_li[:args.N]:
#     p = path.split("/")
#     if ('jpg' in p[-1]): dict[p[1]].append(path)

# #sepalate train:val:test
# dataset_dict = defaultdict(list)
# half = lambda x: (x[:len(x)//2], x[len(x)//2:])

# for k,v in dict.items():
#     len_ = len(v)
#     random.shuffle(v)
#     data_tr, data_ = half(v)
#     data_va, data_te = half(data_)
#     dataset_dict['train']+=data_tr
#     dataset_dict['val']+=data_va
#     dataset_dict['test']+=data_te

# print("The split was successful.\n train:val:test = {}:{}:{}".format(len(dataset_dict['train']),
#     len(dataset_dict['val']),
#     len(dataset_dict['test'])))
# with open(args.out_path, "wb") as f:
#     pickle.dump(dataset_dict, f)
