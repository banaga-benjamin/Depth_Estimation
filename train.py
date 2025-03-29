from time import time

import metrics
import dataset
import synthesis
import constants

from networks import depth_encoder
from networks import depth_decoder
from networks import depth_convgru

from networks import pose_encoder
from networks import pose_decoder

import torch
from pathlib import Path
from torch.nn import functional
from torchvision import transforms
from torch.utils.data import DataLoader


def train_step(dataloader: DataLoader, d_encoder: depth_encoder.DepthEncoder, d_decoder: depth_decoder.DepthDecoder, convgru: depth_convgru.ConvGru,
                    p_encoder: pose_encoder.PoseEncoder, p_decoder: pose_decoder.PoseDecoder, optimizer, device: str = "cpu", sid = False):
    # determine which depths should be used in cost volume
    if sid: DEPTHS = constants.SID_DEPTHS
    else: DEPTHS = constants.UID_DEPTHS

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
            pose_mats = torch.stack([synthesis.construct_pose(p_decoder_output, device) for p_decoder_output in p_decoder_outputs], dim = 0)

            # synthesize images for computing cost volumes
            synthesized_imgs = [
                synthesis.synthesize_from_depths(constants.COST_INTRINSIC_MAT.to(device), constants.COST_INTRINSIC_INV.to(device), pose_mats[idx], DEPTHS,
                                                  constants.COST_WIDTH, constants.COST_HEIGHT, img_seq[idx], device).detach( )
                for idx in range(len(img_seq) - 1)]

            # compute the cost volumes relative to source images
            resized_target_img = functional.interpolate(img_seq[-1].unsqueeze(dim = 0), size = (constants.COST_HEIGHT, constants.COST_WIDTH), mode = "bilinear")
            cost_volumes = metrics.cost_volumes(resized_target_img, torch.stack(synthesized_imgs))

            # obtain depth outputs from depth decoder -> convgru
            depth_outputs = convgru(d_decoder(d_encoder_outputs, cost_volumes))
                
            # synthesize the source images from the target image and final depth predictions
            proj_pixels = torch.stack([
                synthesis.reproject_from_depth(constants.IMG_INTRINSIC_MAT.to(device), constants.IMG_INTRINSIC_INV.to(device), pose_mats[idx], depth_outputs[idx], device = device)
                for idx in range(len(img_seq) - 1)
                ], dim = 0)

            output_imgs = synthesis.synthesize(src_imgs, proj_pixels)

            # update lists used in computation of loss
            target_imgs_list.append(torch.stack([target_img] * (len(img_seq) - 1), dim = 0))
            output_imgs_list.append(output_imgs)
            depth_outputs_list.append(depth_outputs)

        # calculate overall loss
        overall_loss = metrics.regularization_term(torch.cat(depth_outputs_list, dim = 0), torch.cat(target_imgs_list, dim = 0))
        overall_loss += metrics.reprojection_loss(torch.cat(output_imgs_list, dim = 0), torch.cat(target_imgs_list, dim = 0))

        # perform backpropagation
        optimizer.zero_grad( )
        overall_loss.backward( )
        optimizer.step( )

        torch.cuda.empty_cache( )

        if batch % 100 == 0:
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
    print("\nnote: trained models will be saved at 'trained_models' folder")

    train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(num_magnitude_bins = constants.NUM_RANDOM_TRANS),
        transforms.Resize((constants.HEIGHT, constants.WIDTH)),
        transforms.ToTensor( )
    ])

    train_data = dataset.TrainingData(seq_len = constants.SEQ_LEN, device = device, transform = train_transform)
    train_dataloader = DataLoader(train_data, batch_size = constants.BATCH_SIZE, num_workers = constants.NUM_WORKERS // 2,
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

    folder = Path("trained_models")
    folder.mkdir(exist_ok = True)  # create folder if needed

    for epoch in range(constants.EPOCHS):
        print( ); print("-" * 50)
        print("Current Epoch:", epoch + 1, " / ", constants.EPOCHS)
        d_decoder.train( ); p_decoder.train( ); convgru.train( )
        train_step(train_dataloader, d_encoder, d_decoder, convgru, p_encoder, p_decoder, optimizer, device, sid = True)

        # save model weights
        torch.save(convgru.state_dict( ), folder / f"convgru_weights_{epoch}.pth")
        torch.save(d_decoder.state_dict( ), folder / f"depth_decoder_weights_{epoch}.pth")
        torch.save(p_decoder.state_dict( ), folder / f"pose_decoder_weights_{epoch}.pth")
