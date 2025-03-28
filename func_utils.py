import torch
import numpy as np
from torch import nn
from torch.nn import functional
from torchvision import transforms


def reproject_from_depth(pose_mat, depth_map, width = 640, height = 192, device = "cpu", intrinsic_mat = None, intrinsic_inv = None):
    # intrinsic matrices are specified as 4 x 4 matrices
    # suitable for homogenous coordinate transformations

    # if no intrinsic matrices are supplied
    if intrinsic_mat is None:
        intrinsic_mat = torch.Tensor(
            [[0.58, 0.00, 0.50, 0.00],
            [0.00, 1.92, 0.50, 0.00],
            [0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.00, 1.00]]
        ).to(device)

        intrinsic_mat[0, :] *= width; intrinsic_mat[1, :] *= height
        intrinsic_inv = torch.linalg.pinv(intrinsic_mat)

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
    if depth_map.dim( ) > 3: depth_map = depth_map.squeeze( )
    if depth_map.dim( ) > 2: depth_map = depth_map.squeeze( )
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

    depths = np.arange(0, 80, 80 / depths)

    # for each depth in supplied depths
    for depth in depths:
        # create a uniform depth map
        depth_map = torch.ones(height, width, device = device) * depth

        # project some target image to the source image
        proj_pixels = reproject_from_depth(pose_mat, depth_map, width, height, device, intrinsic_mat, intrinsic_inv)

        # synthesize the target image
        synthesized_img = synthesize(src_img, proj_pixels)

        # synthesized img should be of dimensions (C, H, W)
        if synthesized_img.dim( ) > 3: synthesized_img = torch.squeeze(synthesized_img)
        synthesized_imgs.append(synthesized_img)
    return torch.stack(synthesized_imgs)


def cost_volume(target_img, synthesized_imgs):
    costs = list( )

    # target image should be of dimension (C, H, W)
    if target_img.dim( ) > 3: target_img = target_img.squeeze( )
    target_imgs = torch.stack([target_img] * len(synthesized_imgs))

    # get the mean of the differences along dimension C
    costs = (target_imgs - synthesized_imgs).mean(dim = 1)
    return costs


def construct_pose(pose_vector, device):
    pose_vector = torch.squeeze(pose_vector)
    rot_vec = pose_vector[:3]
    trans_vec = pose_vector[3:].view(-1, 1)

    theta = torch.linalg.norm(rot_vec); u = rot_vec / theta
    if theta < 1e-10:  rot_mat = torch.eye(3)
    else:
        U = torch.Tensor([[0, -u[2], u[1]],
                          [u[2], 0, -u[0]],
                          [-u[1], u[0], 0]]).to(device)
        rot_mat = torch.eye(3).to(device) + torch.sin(theta) * U + (1 - torch.cos(theta)) * torch.matmul(U, U)
    pose_mat = torch.cat((rot_mat, trans_vec), dim = 1)
    return pose_mat
    

def reprojection_loss(preds, targets):
    if targets.dim( ) < 4: targets = targets.unsqueeze(dim = 0)
    if preds.dim( ) < 4: preds = preds.unsqueeze(dim = 0)

    x = functional.pad(preds, pad = (1, 1, 1, 1)); y = functional.pad(targets, pad = (1, 1, 1, 1))

    mu_x = functional.avg_pool2d(x, kernel_size = 3, stride = 1)
    mu_y = functional.avg_pool2d(y, kernel_size = 3, stride = 1)

    sigma_x  = functional.avg_pool2d(x ** 2, kernel_size = 3, stride = 1) - mu_x ** 2
    sigma_y  = functional.avg_pool2d(y ** 2, kernel_size = 3, stride = 1) - mu_y ** 2
    sigma_xy = functional.avg_pool2d(x * y, kernel_size = 3, stride = 1) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + (0.01 ** 2)) * (2 * sigma_xy + (0.03 ** 2))
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + (0.01 ** 2)) * (sigma_x + sigma_y + (0.03 ** 2))

    ssim_loss = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1).mean(dim = 1, keepdim = True)
    l1_loss = torch.abs(targets - preds).mean(dim = 1, keepdim = True)

    # return 0.05 * ssim_loss + 0.15 * l1_loss
    return (0.05 * ssim_loss + 0.15 * l1_loss).mean( )


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