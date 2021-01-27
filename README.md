# cUNet-Pytorch
This is pytorch implementation of conditional U-Net.
You can train or test conditional image transformation with semi-supervised learning.
![onehotss](./docs/onehot.png)
![softss](./docs/semisupervised.png)
- Paper: coming soon
- Pre-trained model: coming soon

- distributed train command
`python -m torch.distributed.launch --nproc_per_node=[NUM_GPUS] t_[ANY_TRAIN_CODE].py --amp args []`