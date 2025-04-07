import metrics
import constants
import synthesis

from networks import posenet
from networks import depth_encoder
from networks import depth_decoder
from networks import depth_convgru

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torch.nn import functional
import torchvision.transforms.functional as tf


def predict(image_data: list[str], encoder: depth_encoder.DepthEncoder, decoder: depth_decoder.DepthDecoder, convgru: depth_convgru.ConvGru,
            pose: posenet.PoseNet, device: str = "cpu") -> None:
    for idx in range(len(image_data) - constants.SEQ_LEN + 1):
        print("processed:", image_data[idx + constants.SEQ_LEN - 1])    # pseudo log

        # collect input images and corresponding file names
        img_seq = list( ); img_seq_path = list( )
        for img_path in image_data[idx: idx + constants.SEQ_LEN]:
            img_seq_path.append(str(img_path)[:-4])

            img = Image.open(img_path)
            img = tf.to_tensor(tf.resize(img, size = (192, 640)))
            img_seq.append(img)
        img_seq = torch.stack(img_seq)
        img_seq = img_seq.to(device)

        # target image has dimensions (C, H, W)
        # image sequence has dimensions (N, C, H, W)
        target_img = img_seq[-1]; src_imgs = img_seq[:-1]

        # obtain pose matrix relative to source images
        pose_net_outputs = pose(torch.stack([target_img] * (constants.SEQ_LEN - 1)), src_imgs)
        pose_mats = synthesis.construct_pose_batch(pose_net_outputs, device)

        # synthesize images for computing cost volumes
        synthesized_imgs = [
            synthesis.synthesize_from_depths(constants.COST_INTRINSIC_MAT.to(device), constants.COST_INTRINSIC_INV.to(device), pose_mats[idx],
                                                constants.DEPTHS, constants.COST_WIDTH, constants.COST_HEIGHT, img_seq[idx], device)
            for idx in range(constants.SEQ_LEN - 1)]

         # compute the cost volumes relative to source images
        resized_target_img = functional.interpolate(img_seq[-1].unsqueeze(dim = 0), size = (constants.COST_HEIGHT, constants.COST_WIDTH), mode = "bilinear")
        cost_volumes = torch.stack([metrics.cost_volumes(resized_target_img, synthesized_img) for synthesized_img in synthesized_imgs])

        # obtain candidate depths from cost volumes
        candidate_depths = synthesis.depth_from_costs(cost_volumes, num_channels = constants.COST_DEPTHS, sid = constants.USE_SID, device = device)

       # obtain depth encoder outputs
        encoder_output = encoder(target_img.unsqueeze(dim = 0))

        # obtain final depths by passing to depth decoder -> convgru
        depth_outputs = list( )
        for candidate_depth in candidate_depths:
            depth_outputs.append(convgru(decoder(encoder_output, candidate_depth.unsqueeze(dim = 0))))
        depth_outputs = torch.cat(depth_outputs, dim = 0)
        
        # get last depth output and convert to numpy
        depth = depth_outputs[-1].squeeze( ).cpu( ).numpy( )

        # normalize to [0, 255]
        depth = depth * 255
        depth = depth.astype(np.uint8)

        # save depth map
        plt.imsave(img_seq_path[-1] + "_depth.png", depth, cmap = "cool")


if __name__ == "__main__":
    print("pytorch version:", torch.__version__)

    device = 'cuda' if torch.cuda.is_available( ) else 'cpu'
    print("using device:", device)

    # initialize networks
    pose = posenet.PoseNet( ).to(device)
    convgru = depth_convgru.ConvGru( ).to(device)
    encoder = depth_encoder.DepthEncoder( ).to(device)
    decoder = depth_decoder.DepthDecoder( ).to(device)

    # get input for loading trained models
    print("\nload trained model? [Y/N]: ", end = "")
    load_train = input( )

    # load trained models
    if load_train == "Y":
        print("input path to trained pose net: ", end = "")
        pose.load_state_dict(torch.load(input( )))

        print("input path to trained depth decoder: ", end = "")
        decoder.load_state_dict(torch.load(input( )))
        
        print("input path to trained convgru model: ", end = "")
        convgru.load_state_dict(torch.load(input( )))
    print( )

    # get path to image data
    print("input path to image data folder: ", end = "")
    data_directory = Path(input( ))

    # collect image inputs in folder
    image_data = list( )
    for img in data_directory.iterdir( ):
        if img.suffix != ".png": continue
        image_data.append(img)
    
    # make predictions
    pose.eval( ); decoder.eval( ); convgru.eval( )
    with torch.no_grad( ): predict(image_data, encoder, decoder, convgru, pose, device)
