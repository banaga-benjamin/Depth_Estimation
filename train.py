from time import time

import metrics
import dataset
import synthesis
import constants

from networks import posenet
from networks import depth_encoder
from networks import depth_decoder
from networks import depth_convgru

import torch
from pathlib import Path
from torch.nn import utils
from itertools import chain
from torch.nn import functional
from torch.utils.data import DataLoader


def train_step(dataloader: DataLoader, encoder: depth_encoder.DepthEncoder, decoder: depth_decoder.DepthDecoder,  convgru: depth_convgru.ConvGru,
               pose: posenet.PoseNet, optimizer, device: str = "cpu") -> None:
    """
        trains depth encoder, decoder, and convgru networks and pose network \n
        input: dataloader, depth encoder, decoder, and convgru networks and pose network, an optimizer, and device to store tensors \n
        output: trained networks saved at 'trained_models' folder
    """

    cumulative_loss = 0
    start_time = time( )
    num_batches = len(dataloader)
    print("Number of Batches:", num_batches, "\n")
    for batch, img_batch in enumerate(dataloader):
        # create list of depths for logging
        depth_outputs_list = list( )

        # initialize loss
        overall_loss = 0
        for img_seq in img_batch:
            # target image has dimensions (C, H, W)
            # image sequence has dimensions (N, C, H, W)
            target_img = img_seq[-1]; src_imgs = img_seq[:-1]

            # obtain pose matrix relative to source images
            pose_net_outputs = pose(torch.stack([target_img] * (constants.SEQ_LEN - 1)), src_imgs)
            pose_mats = synthesis.construct_pose_batch(pose_net_outputs, device)

            # synthesize images for computing cost volumes
            synthesized_imgs = [
                synthesis.synthesize_from_depths(constants.COST_INTRINSIC_MAT.to(device), constants.COST_INTRINSIC_INV.to(device), pose_mat,
                                                 constants.DEPTHS, constants.COST_WIDTH, constants.COST_HEIGHT, src_img, device).detach( )
                for pose_mat, src_img in zip(pose_mats, src_imgs)]
            
            # compute the cost volumes relative to source images
            resized_target_img = functional.interpolate(img_seq[-1].unsqueeze(dim = 0), size = (constants.COST_HEIGHT, constants.COST_WIDTH), mode = "bilinear")
            cost_volumes = torch.stack([metrics.cost_volumes(resized_target_img, synthesized_img) for synthesized_img in synthesized_imgs])

            # obtain candidate depths from cost volumes
            candidate_depths = synthesis.depth_from_costs(cost_volumes, num_channels = constants.COST_DEPTHS, sid = constants.USE_SID, device = device)
            candidate_depths = functional.interpolate(candidate_depths, size = (constants.HEIGHT, constants.WIDTH), mode = "bicubic")

            # obtain depth encoder outputs
            encoder_output = encoder(target_img.unsqueeze(dim = 0))

            # obtain final depths by passing to depth decoder -> convgru
            depth_outputs = list( )
            for candidate_depth in candidate_depths:
                depth_outputs.append(convgru(decoder(encoder_output, candidate_depth.unsqueeze(dim = 0))))
            depth_outputs = torch.cat(depth_outputs, dim = 0) * constants.MAX_DEPTH

            # synthesize the target image from the source images and final depth predictions
            proj_pixels = torch.stack([
                synthesis.reproject_from_depth(constants.IMG_INTRINSIC_MAT.to(device), constants.IMG_INTRINSIC_INV.to(device),
                                               pose_mat, depth_output, device = device)
                for pose_mat, depth_output in zip(pose_mats, depth_outputs)])

            output_imgs = synthesis.synthesize(src_imgs, proj_pixels)

            # stack target images for alignment
            target_imgs = torch.stack([target_img] * (constants.SEQ_LEN - 1), dim = 0)

            # compute loss values
            regularization_term = metrics.regularization_term(depth_outputs / constants.MAX_DEPTH, target_imgs)
            output_reprojection = metrics.reprojection_loss(output_imgs, target_imgs)
            src_reprojection = metrics.reprojection_loss(src_imgs, target_imgs)

            # update overall loss
            overall_loss += torch.mean(torch.mean(output_reprojection / torch.exp(src_reprojection + 1e-6), dim = (2, 3)))
            overall_loss += 0.05 * regularization_term  # 0.01 works, try increasing

            # for logging
            cumulative_loss += torch.mean(torch.mean(output_reprojection, dim = (1, 2, 3))).item( )
            depth_outputs_list.append(depth_outputs)

        # log updates
        if batch % 500 == 0 and batch != 0:
            print("max depth:", torch.max(torch.stack(depth_outputs_list)).item( ))
            print("average depth:", torch.mean(torch.stack(depth_outputs_list)).item( ))
            print("average loss:", cumulative_loss / (batch + 1) / constants.BATCH_SIZE); print( )

        # perform backpropagation
        optimizer.zero_grad( )
        overall_loss.backward( )

        # clip gradients to avoid exploding gradients
        utils.clip_grad_norm_(decoder.parameters( ), max_norm = 1.0)
        utils.clip_grad_norm_(convgru.parameters( ), max_norm = 1.0)
        utils.clip_grad_norm_(pose.parameters( ), max_norm = 1.0)
        optimizer.step( )

        torch.cuda.empty_cache( )

        if batch % 1000 == 0 and batch != 0:
            elapsed_time = (time( ) - start_time)

            print("batches completed:", batch)
            print("time remaining:", (elapsed_time / (batch + 1)) * (num_batches - batch - 1) / 60 , "minutes")
            print("-" * 50, "\n")
    print("\n" + "time elapsed:", elapsed_time / 60, "minutes")


if __name__ == "__main__":
    print("pytorch version:", torch.__version__)

    device = 'cuda' if torch.cuda.is_available( ) else 'cpu'
    print("using device:", device)
    print("\n" + "note: trained models will be saved at 'trained_models' folder")

    # load training data
    train_data = dataset.TrainingData(seq_len = constants.SEQ_LEN, device = device)
    train_dataloader = DataLoader(train_data, batch_size = constants.BATCH_SIZE, num_workers = constants.NUM_WORKERS, shuffle = True, drop_last = True)
    
    # initialize networks
    pose = posenet.PoseNet( ).to(device)
    convgru = depth_convgru.ConvGru( ).to(device)
    encoder = depth_encoder.DepthEncoder( ).to(device)
    decoder = depth_decoder.DepthDecoder( ).to(device)

    # create optimizer
    optimizer = torch.optim.Adam(chain(pose.parameters( ), decoder.parameters( ), convgru.parameters( )), lr = 1e-4, weight_decay = 1e-5)

    # location to save trained networks
    folder = Path("trained_models")
    folder.mkdir(exist_ok = True)  # create folder if needed

    # perform training loop
    for epoch in range(constants.EPOCHS):
        print( ); print("-" * 50)
        print("Current Epoch:", epoch + 1, " / ", constants.EPOCHS)

        # train networks
        pose.train( ); decoder.train( ); convgru.train( )
        train_step(train_dataloader, encoder, decoder, convgru, pose, optimizer, device)

        # save model weights
        torch.save(convgru.state_dict( ), folder / f"convgru_{epoch}.pth")
        torch.save(decoder.state_dict( ), folder / f"decoder_{epoch}.pth")
        torch.save(pose.state_dict( ), folder / f"pose_{epoch}.pth")
