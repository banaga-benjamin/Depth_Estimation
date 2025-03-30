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


def test_step(dataloader: DataLoader, d_encoder: depth_encoder.DepthEncoder, d_decoder: depth_decoder.DepthDecoder, convgru: depth_convgru.ConvGru,
                    p_encoder: pose_encoder.PoseEncoder, p_decoder: pose_decoder.PoseDecoder, device: str = "cpu"):
    # determine which depths should be used in cost volume
    if constants.USE_SID: DEPTHS = constants.SID_DEPTHS
    else: DEPTHS = constants.UID_DEPTHS

    # initialize error metrics
    cumulative_rmse = 0
    cumulative_rmsle = 0
    cumulative_sq_rel = 0
    cumulative_abs_rel = 0


    start_time = time( )
    num_batches = len(dataloader.dataset) // dataloader.batch_size
    print("Number of Batches:", num_batches, "\n")
    for batch, test_batch in enumerate(dataloader):
        # create list of depth outputs
        depth_outputs_list = list( )

        img_batch, ground_truth_batch = test_batch
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
            depth_outputs_list.append(depth_outputs)

        # preprocess ground truths to align shape with depth outputs
        ground_truths_list = list( )
        for idx in range(constants.BATCH_SIZE):
            ground_truths_list.append(torch.stack([ground_truth_batch[idx]] * (constants.SEQ_LEN - 1)))

        # convert depth outputs list and ground truths list to tensors
        ground_truth_stacked = torch.cat(ground_truths_list, dim = 0)
        depth_outputs_stacked = torch.cat(depth_outputs_list, dim = 0)

        # accumulate error metrics
        cumulative_rmse += metrics.rmse(depth_outputs_stacked, ground_truth_stacked)
        cumulative_rmsle += metrics.rmsle(depth_outputs_stacked, ground_truth_stacked)
        cumulative_sq_rel += metrics.sq_rel(depth_outputs_stacked, ground_truth_stacked)
        cumulative_abs_rel += metrics.abs_rel(depth_outputs_stacked, ground_truth_stacked)


        if batch % 100 == 0:
            if batch == 0: continue
            elapsed_time = (time( ) - start_time)

            print("batches completed:", batch)
            print("time remaining:", (elapsed_time / (batch + 1)) * (num_batches - batch - 1) / 60 , "minutes")
            print("-" * 50, "\n")
    
    # print averages of accumulated error metrics
    print( ); print("-" * 50)
    print("RMSE:\t", (cumulative_rmse / num_batches).item( ) * constants.MAX_DEPTH)     # RMSE is not scale independent
    print("RMSLE:\t", (cumulative_rmsle / num_batches).item( ))
    print("Sq Rel:\t", (cumulative_sq_rel / num_batches).item( ))
    print("Abs Rel:\t", (cumulative_abs_rel / num_batches).item( ))
    print("-" * 50)

    print("\ntime elapsed:", elapsed_time / 60, "minutes")


if __name__ == "__main__":
    print("pytorch version:", torch.__version__)

    device = 'cuda' if torch.cuda.is_available( ) else 'cpu'
    print("using device:", device)

    test_transform = transforms.Compose([
        transforms.Resize((constants.HEIGHT, constants.WIDTH)),
        transforms.ToTensor( )
    ])

    test_data = dataset.TestingData(seq_len = constants.SEQ_LEN, device = device, transform = test_transform)
    test_dataloader = DataLoader(test_data, batch_size = constants.BATCH_SIZE, num_workers = constants.NUM_WORKERS // 2, drop_last = True)
    
    convgru = depth_convgru.ConvGru( ).to(device)
    d_encoder = depth_encoder.DepthEncoder( ).to(device)
    d_decoder = depth_decoder.DepthDecoder( ).to(device)

    p_encoder = pose_encoder.PoseEncoder( ).to(device)
    p_decoder = pose_decoder.PoseDecoder(device = device).to(device)

    # get input for loading trained models
    print("load trained model? [Y/N]: ", end = "")
    load_train = input( )

    # load trained models
    if load_train == "Y":
        print("Input path to trained pose decoder: ", end = "")
        p_decoder.load_state_dict(torch.load(input( )))
        
        print("Input path to trained depth decoder: ", end = "")
        d_decoder.load_state_dict(torch.load(input( )))
        
        print("Input path to trained convgru model: ", end = "")
        convgru.load_state_dict(torch.load(input( )))

    d_decoder.eval( ); p_decoder.eval( ); convgru.eval( )
    with torch.no_grad( ): test_step(test_dataloader, d_encoder, d_decoder, convgru, p_encoder, p_decoder, device)
