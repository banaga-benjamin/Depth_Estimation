import torch
import numpy as np
from torch.nn import functional

def reproject(intrinsic_mat, intrinsic_inv, pose_mat, depth_map, width, height, device):
    # intrinsic matrices are specified as 4 x 4 matrices
    # suitable for homogenous coordinate transformations

    # create a matrix of 4d pixel coordinates
    x_coords = torch.arange(width, dtype = torch.float32, device = device)
    y_coords = torch.arange(height, dtype = torch.float32, device = device)
    output_pixels = torch.stack([
        x_coords.view(1, -1).expand(height, width),
        y_coords.view(-1, 1).expand(height, width),
        torch.ones(height, width, dtype = torch.float32, device = device),
        torch.ones(height, width, dtype = torch.float32, device = device)
    ], dim = 2)

    # multiply 4d pixel coordinates by inverse intrinsic matrix
    output_pixels = torch.matmul(output_pixels, intrinsic_inv.T)

    # convert depth map to a matrix that can be multiplied with output pixels
    depth_map = torch.stack([
        depth_map, depth_map, depth_map, torch.ones(height, width, device = device)
    ], dim = 2)

    # multiply (element-wise) 4d pixel coordinates by converted depth map
    output_pixels = output_pixels * depth_map

    # multiply 4d pixel coordinates by 6-DOF pose matrix
    output_pixels = torch.matmul(output_pixels, pose_mat.T)

    # multiply 3d pixel coordinates by intrinsic matrix
    output_pixels = torch.matmul(output_pixels, intrinsic_mat[:3, :3].T)

    # convert 3d homogenous pixel coordinates to 2d pixel coordinates
    x = output_pixels[..., 0]; y = output_pixels[..., 1]; z = output_pixels[..., 2]
    z = torch.where(z == 0, torch.ones_like(z), z)
    output_pixels = torch.stack([x / z, y / z], dim = 2)

    return output_pixels


def synthesize(src_img, proj_pixels):
    # source image dimension should be (N, C, H, W)
    # projected pixel dimensioon should be (N, H, W, 2)
    if src_img.dim( ) < 4: src_img = src_img.unsqueeze(dim = 0)
    if proj_pixels.dim( ) < 4: proj_pixels = proj_pixels.unsqueeze(dim = 0)

    # synthesize some target image using source image and pixels from the target image project to source image
    return functional.grid_sample(input = src_img, grid = proj_pixels, padding_mode = "border", align_corners = True)


def synthesize_from_depths(intrinsic_mat, intrinsic_inv, pose_mat, depths, width, height, src_img, device):
    synthesized_imgs = list( )

    # for each depth in supplied depths
    for depth in depths:
        # create a uniform depth map
        depth_map = torch.ones(height, width, device = device) * depth

        # project some target image to the source image
        proj_pixels = reproject(intrinsic_mat, intrinsic_inv, pose_mat, depth_map, width, height, device)

        # synthesize the target image
        synthesized_img = synthesize(src_img, proj_pixels)

        synthesized_imgs.append(synthesized_img)
    return synthesized_imgs


def cost_volume(target_img, synthesized_imgs):
    costs = list( )

    # target image dimension should be (C, H, W)
    if target_img.dim( ) > 3: target_img.squeeze( )
    for synthesized_img in synthesized_imgs:
        # synthesized image dimension should be (C, H, W)
        if synthesized_img.dim( ) > 3: synthesized_img.squeeze( )

        # take the per-pixel difference and get the mean difference
        # per pixel across the dimension of the channels
        costs.append((target_img - synthesized_img).mean(dim = 0))
    costs = torch.stack(costs, dim = 0)
    return costs


# for debugging
# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available( ) else 'cpu'
#     width = 8; height = 4

#     intrinsic_mat = np.array(
#             [[0.58, 0.00, 0.50, 0.00],
#             [0.00, 1.92, 0.50, 0.00],
#             [0.00, 0.00, 1.00, 0.00],
#             [0.00, 0.00, 0.00, 1.00]],
#             dtype = np.float32
#         )
    
#     intrinsic_mat[0, :] *= width
#     intrinsic_mat[1, :] *= height

#     intrinsic_inv = np.linalg.pinv(intrinsic_mat)

#     pose_mat = torch.rand(3, 4, device = device)
#     depth_map = torch.rand(height, width, device = device) * 80

#     intrinsic_mat = torch.from_numpy(intrinsic_mat).to(device)
#     intrinsic_inv = torch.from_numpy(intrinsic_inv).to(device)

#     proj_pixels = reproject(intrinsic_mat, intrinsic_inv, pose_mat, depth_map, width, height, device)
#     print(proj_pixels.shape)

#     src_img = torch.rand(3, height, width, device = device)
#     synthesized_img = synthesize(src_img, proj_pixels)
#     print(synthesized_img.shape)

#     target_img = torch.rand(3, height, width, device = device)
#     costs = cost_volume(target_img, synthesized_img)
#     for cost in costs: print(cost.shape)
