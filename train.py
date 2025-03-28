from time import time
from os import cpu_count

import dataset
import func_utils
from networks import depth_encoder
from networks import depth_decoder
from networks import depth_convgru

from networks import pose_encoder
from networks import pose_decoder

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader


def train_step(dataloader: DataLoader, d_encoder: depth_encoder.DepthEncoder, d_decoder: depth_decoder.DepthDecoder, convgru: depth_convgru.ConvGru,
                    p_encoder: pose_encoder.PoseEncoder, p_decoder: pose_decoder.PoseDecoder, device: str = "cpu"):
    # set height, width, and depths
    # for cost volume calculation
    H = 192; W = 640
    depths = 64; cost_height = H // 2; cost_width = W // 2

    # calculate intrinsic matrix for cost volume
    intrinsic_mat = torch.Tensor(
            [[0.58, 0.00, 0.50, 0.00],
            [0.00, 1.92, 0.50, 0.00],
            [0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.00, 1.00]]
        ).to(device)

    intrinsic_mat[0, :] *= cost_width; intrinsic_mat[1, :] *= cost_height
    intrinsic_inv = torch.linalg.pinv(intrinsic_mat)

    start_time = time( )
    num_batches = len(dataloader.dataset) // dataloader.batch_size
    print("Number of Batches:", num_batches, "\n")
    for batch, img_batch in enumerate(dataloader):
        for img_seq in img_batch:
            # img_seq dimension is (N, C, H, W)
            # single img dimension is (C, H, W)

            d_encoder_outputs = d_encoder(img_seq[-1])
            for idx in range(len(img_seq) - 1):
                p_encoder_outputs = p_encoder(torch.stack((img_seq[-1], img_seq[idx])))
                p_decoder_output = p_decoder(p_encoder_outputs[0], p_encoder_outputs[1])
                pose_mat = func_utils.construct_pose(p_decoder_output, device)

                synthesized_imgs = func_utils.synthesize_from_depths(intrinsic_mat, intrinsic_inv, pose_mat, depths, cost_width, cost_height, img_seq[idx], device)

                resize_img = transforms.Resize(size = (cost_height, cost_width))
                cost_volume = func_utils.cost_volume(resize_img(img_seq[-1]), synthesized_imgs)
                depth_outputs = convgru(d_decoder(d_encoder_outputs, cost_volume))
        print("batch no.", batch, "completed")
        torch.cuda.empty_cache( )

        if batch % 100 == 0:
            if batch == 0: continue
            stop_time = time( )
            elapsed_time = (stop_time - start_time)

            print("batches completed:", batch)
            print("time remaining:", (elapsed_time / (batch + 1)) * (num_batches - batch - 1) / 60 , "minutes")
            print("-" * 50, "\n")
    print("\ntime elapsed:", elapsed_time / 60, "minutes")


if __name__ == "__main__":
    print("pytorch version:", torch.__version__)

    device = 'cuda' if torch.cuda.is_available( ) else 'cpu'
    print("using device:", device)

    train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(num_magnitude_bins = 7),
        transforms.Resize((192, 640)),
        transforms.ToTensor( )
    ])

    BATCH_SIZE = 6; NUM_WORKERS = cpu_count( )
    train_data = dataset.TrainingData(seq_len = 4, device = device, transform = train_transform)
    train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS // 2,
                                  persistent_workers = True, shuffle = True, drop_last = True)
    
    convgru = depth_convgru.ConvGru(device = device)
    d_encoder = depth_encoder.DepthEncoder(device = device)
    d_decoder = depth_decoder.DepthDecoder(device = device)

    p_encoder = pose_encoder.PoseEncoder(device = device)
    p_decoder = pose_decoder.PoseDecoder(device = device)

    EPOCHS = 1; print("Number of Epochs:", EPOCHS, "\n")
    for epoch in range(EPOCHS):
        print("Current Epoch:", epoch)
        with torch.no_grad( ): train_step(train_dataloader, d_encoder, d_decoder, convgru, p_encoder, p_decoder, device)
