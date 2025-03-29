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
from torch.nn import functional
from torchvision import transforms
from torch.utils.data import DataLoader


def train_step(dataloader: DataLoader, d_encoder: depth_encoder.DepthEncoder, d_decoder: depth_decoder.DepthDecoder, convgru: depth_convgru.ConvGru,
                    p_encoder: pose_encoder.PoseEncoder, p_decoder: pose_decoder.PoseDecoder, optimizer, device: str = "cpu"):
    # set height, width, and depths for cost volume calculation
    cost_depths = 64; cost_height = 192 // 2; cost_width = 640 // 2

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
        # create lists for computation of loss
        target_imgs_list = list( )
        output_imgs_list = list( )
        depth_outputs_list = list( )

        for img_seq in img_batch:
            # target image has dimensions (C, H, W)
            # image sequence has dimensions (N, C, H, W)
            target_img = img_seq[-1]; src_imgs = img_seq[:-1]

            # obtain depth encoder output from target image
            d_encoder_outputs = d_encoder(target_img)

            # obtain pose matrix relative to source images
            p_encoder_outputs = p_encoder(img_seq)
            p_decoder_outputs = p_decoder(p_encoder_outputs[-1], p_encoder_outputs[:-1])
            pose_mats = torch.stack([func_utils.construct_pose(p_decoder_output, device) for p_decoder_output in p_decoder_outputs], dim = 0)

            # synthesize images for computing cost volumes
            synthesized_imgs = [
                func_utils.synthesize_from_depths(intrinsic_mat, intrinsic_inv, pose_mats[idx], cost_depths,
                                                  cost_width, cost_height, img_seq[idx], device).detach( )
                for idx in range(len(img_seq) - 1)]

            # compute the cost volumes relative to source images
            resized_target_img = functional.interpolate(img_seq[-1].unsqueeze(dim = 0), size = (cost_height, cost_width), mode = "bilinear")
            cost_volumes = func_utils.cost_volumes(resized_target_img, torch.stack(synthesized_imgs))

            # obtain depth outputs from depth decoder -> convgru
            depth_outputs = convgru(d_decoder(d_encoder_outputs, cost_volumes))
                
            # synthesize the source images from the target image and final depth predictions
            proj_pixels = torch.stack([
                func_utils.reproject_from_depth(pose_mats[idx], depth_outputs[idx], device = device)
                for idx in range(len(img_seq) - 1)
                ], dim = 0)

            output_imgs = func_utils.synthesize(src_imgs, proj_pixels)

            # update lists used in computation of loss
            target_imgs_list.append(torch.stack([target_img] * (len(img_seq) - 1), dim = 0))
            output_imgs_list.append(output_imgs)
            depth_outputs_list.append(depth_outputs)

        # calculate overall loss
        overall_loss = func_utils.regularization_term(torch.cat(depth_outputs_list, dim = 0), torch.cat(target_imgs_list, dim = 0))
        overall_loss += func_utils.reprojection_loss(torch.cat(output_imgs_list, dim = 0), torch.cat(target_imgs_list, dim = 0))

        # perform backpropagation
        optimizer.zero_grad( )
        overall_loss.backward( )
        optimizer.step( )

        torch.cuda.empty_cache( )

        if batch % 10 == 0:
            if batch == 0: continue
            elapsed_time = (time( ) - start_time)

            print("batches completed:", batch)
            print("time remaining:", (elapsed_time / (batch + 1)) * (num_batches - batch - 1) / 60 , "minutes")
            print("-" * 30, "\n")
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

    BATCH_SIZE = 8; NUM_WORKERS = cpu_count( )
    train_data = dataset.TrainingData(seq_len = 4, device = device, transform = train_transform)
    train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS // 2,
                                  shuffle = True, drop_last = True)
    
    convgru = depth_convgru.ConvGru( ).to(device)
    d_encoder = depth_encoder.DepthEncoder( ).to(device)
    d_decoder = depth_decoder.DepthDecoder( ).to(device)

    p_encoder = pose_encoder.PoseEncoder( ).to(device)
    p_decoder = pose_decoder.PoseDecoder(device = device).to(device)

    optimizer = torch.optim.SGD([
        {'params': convgru.parameters( ), 'lr': 1e-3},
        {'params': d_encoder.parameters( ), 'lr': 1e-3},
        {'params': p_decoder.parameters( ), 'lr': 1e-3}
    ])

    EPOCHS = 1; print("Number of Epochs:", EPOCHS, "\n")
    for epoch in range(EPOCHS):
        print("Current Epoch:", epoch + 1)
        d_decoder.train( ); p_decoder.train( ); convgru.train( )
        train_step(train_dataloader, d_encoder, d_decoder, convgru, p_encoder, p_decoder, optimizer, device)
