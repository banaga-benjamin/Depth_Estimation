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
from torch.nn import functional
from torch.utils.data import DataLoader


def test_step(dataloader: DataLoader, encoder: depth_encoder.DepthEncoder, decoder: depth_decoder.DepthDecoder, convgru: depth_convgru.ConvGru,
              pose: posenet.PoseNet, device: str = "cpu") -> None:
    """
        evaluates the performance of trained network against ground truth depth maps \n
        input: dataloader, trained depth encoder, decoder, and convgru networks, trained pose network, and device to store tensors \n
        output: performance metrics of the trained model
    """

    # initialize error metrics
    cumulative_rmse = 0
    cumulative_rmsle = 0
    cumulative_sq_rel = 0
    cumulative_abs_rel = 0

    start_time = time( )
    num_batches = len(dataloader)
    print("Number of Batches:", num_batches, "\n")
    for batch, test_batch in enumerate(dataloader):
        # create list of depth outputs
        depth_output_batch = list( )

        img_batch, ground_truth_batch = test_batch
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
                                                 constants.DEPTHS, constants.COST_WIDTH, constants.COST_HEIGHT, src_img, device)
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
            depth_output_batch.append(depth_outputs[-1])

        # note to rescale normalized ground truth to [0, 80]
        ground_truth_batch *= constants.MAX_DEPTH
        depth_output_batch = torch.stack(depth_output_batch, dim = 0)

        # accumulate error metrics
        cumulative_rmse += metrics.rmse(depth_output_batch, ground_truth_batch)
        cumulative_rmsle += metrics.rmsle(depth_output_batch, ground_truth_batch)
        cumulative_sq_rel += metrics.sq_rel(depth_output_batch, ground_truth_batch)
        cumulative_abs_rel += metrics.abs_rel(depth_output_batch, ground_truth_batch)

        if batch % 50 == 0 and batch != 0: # error logs
            print("batch:", batch)
            print("rmse:", (cumulative_rmse / batch).item( ))
            print("rmsle:", (cumulative_rmsle / batch).item( ))
            print("sq rel:", (cumulative_sq_rel / batch).item( ))
            print("abs rel:", (cumulative_abs_rel / batch).item( )); print( )

        if batch % 100 == 0 and batch != 0:
            elapsed_time = (time( ) - start_time)

            print("batches completed:", batch)
            print("time remaining:", (elapsed_time / (batch + 1)) * (num_batches - batch - 1) / 60 , "minutes")
            print("-" * 50, "\n")
    
    # print averages of accumulated error metrics
    print( ); print("-" * 50)
    print("RMSE:\t\t", (cumulative_rmse / num_batches).item( ))
    print("RMSLE:\t\t", (cumulative_rmsle / num_batches).item( ))
    print("Sq Rel:\t\t", (cumulative_sq_rel / num_batches).item( ))
    print("Abs Rel:\t", (cumulative_abs_rel / num_batches).item( ))
    print("-" * 50)

    elapsed_time = (time( ) - start_time)
    print("\ntime elapsed:", elapsed_time / 60, "minutes")


if __name__ == "__main__":
    print("pytorch version:", torch.__version__)

    device = 'cuda' if torch.cuda.is_available( ) else 'cpu'
    print("using device:", device)

    # load testing data
    test_data = dataset.TestingData(seq_len = constants.SEQ_LEN, device = device)
    test_dataloader = DataLoader(test_data, batch_size = constants.BATCH_SIZE, num_workers = constants.NUM_WORKERS, drop_last = True)
    
    # initialize networks
    pose = posenet.PoseNet( ).to(device)
    convgru = depth_convgru.ConvGru( ).to(device)
    encoder = depth_encoder.DepthEncoder( ).to(device)
    decoder = depth_decoder.DepthDecoder( ).to(device)

    # get input for loading trained models
    print("\n" + "load trained model? [Y/N]: ", end = "")
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

    # perform testing
    pose.eval( ); decoder.eval( ); convgru.eval( )
    with torch.no_grad( ): test_step(test_dataloader, encoder, decoder, convgru, pose, device)
