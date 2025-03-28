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
                    p_encoder: pose_encoder.PoseEncoder, p_decoder: pose_decoder.PoseDecoder, optimizers, device: str = "cpu"):
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
        target_imgs = list( )   # store target images for computation of reprojection error
        output_imgs = list( )   # store output images for computation of reprojection error
        depth_outputs = list( ) # store depth outputs for computation of regularization term
        for seq, img_seq in enumerate(img_batch):
            # img_seq dimension is (N, C, H, W)
            # single img dimension is (C, H, W)

            target_imgs.append(list( ))
            output_imgs.append(list( ))

            # obtain depth encoder output from target image
            d_encoder_outputs = d_encoder(img_seq[-1])
            for idx in range(len(img_seq) - 1):
                # obtain pose matrix relative to current source image
                p_encoder_outputs = p_encoder(torch.stack((img_seq[-1], img_seq[idx])))
                p_decoder_output = p_decoder(p_encoder_outputs[0], p_encoder_outputs[1])
                pose_mat = func_utils.construct_pose(p_decoder_output, device)

                # synthesize images for computing cost volume
                synthesized_imgs = func_utils.synthesize_from_depths(intrinsic_mat, intrinsic_inv, pose_mat, depths,
                                                                     cost_width, cost_height, img_seq[idx], device).detach( )

                # compute the cost volume relative to source image
                resized_target_img = functional.interpolate(img_seq[-1].unsqueeze(dim = 0), size = (cost_height, cost_width), mode = "bilinear")
                cost_volume = func_utils.cost_volume(resized_target_img, synthesized_imgs).detach( )

                # obtain depth output from depth decoder -> convgru
                depth_output = convgru(d_decoder(d_encoder_outputs, cost_volume))

                # aggregate depth outputs by resizingg to (192, 640) and taking the average of the sum of depths
                for idx in range(len(depth_output)): 
                    depth_output[idx] = functional.interpolate(depth_output[idx], size = (192, 640), mode = "bilinear")
                depth_output = torch.cat(depth_output, dim = 0).mean(dim = 0)
                depth_outputs.append(depth_output)
                
                # synthesize the source image from target image and final depth prediction
                proj_pixels = func_utils.reproject_from_depth(pose_mat, depth_output, device = device)
                output_img = func_utils.synthesize(img_seq[idx], proj_pixels)

                output_imgs[seq].append(output_img.squeeze( ))
                target_imgs[seq].append(img_seq[-1])
            output_imgs[seq] = torch.stack(output_imgs[seq], dim = 0)
            target_imgs[seq] = torch.stack(target_imgs[seq], dim = 0)

        # set optimizer gradients to zero
        for optimizer in optimizers: optimizer.zero_grad( )

        # calculate overall loss
        overall_loss = func_utils.regularization_term(torch.cat(depth_outputs, dim = 0), torch.cat(target_imgs, dim = 0))
        overall_loss += func_utils.reprojection_loss(torch.cat(output_imgs, dim = 0), torch.cat(target_imgs, dim = 0))
        overall_loss.backward( )

        # optimize network weights
        for optimizer in optimizers: optimizer.step( )

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
    
    convgru = depth_convgru.ConvGru(device = device)
    d_encoder = depth_encoder.DepthEncoder(device = device)
    d_decoder = depth_decoder.DepthDecoder(device = device)
    torch.compile(convgru); torch.compile(d_encoder); torch.compile(d_decoder)

    p_encoder = pose_encoder.PoseEncoder(device = device)
    p_decoder = pose_decoder.PoseDecoder(device = device)
    torch.compile(p_encoder); torch.compile(p_decoder)
    
    convgru_parameters = list( )
    for name, param in convgru.named_parameters( ):
        if param.requires_grad: convgru_parameters.append(param)

    d_decoder_parameters = list( )
    for layer in d_decoder.layers:
        for name, param in layer.named_parameters( ):
            if param.requires_grad: d_decoder_parameters.append(param)

    p_decoder_parameters = list( )
    for layer in p_decoder.layers:
        for name, param in layer.named_parameters( ):
            if param.requires_grad: p_decoder_parameters.append(param)

    optimizers = list( )
    optimizers.append(torch.optim.SGD(params = convgru_parameters, lr = 0.01))
    optimizers.append(torch.optim.SGD(params = d_decoder_parameters, lr = 0.01))
    optimizers.append(torch.optim.SGD(params = p_decoder_parameters, lr = 0.01))

    EPOCHS = 1; print("Number of Epochs:", EPOCHS, "\n")
    for epoch in range(EPOCHS):
        print("Current Epoch:", epoch + 1)
        d_decoder.train( ); p_decoder.train( ); convgru.train( )
        train_step(train_dataloader, d_encoder, d_decoder, convgru, p_encoder, p_decoder, optimizers, device)
