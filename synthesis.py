import torch
from numpy import arange
from torch.nn import functional


def construct_pose(pose_vector, device):
    # flatten pose vector
    pose_vector = torch.squeeze(pose_vector)

    # obtain rotation and translation vectors
    rot_vec = pose_vector[:3]
    trans_vec = pose_vector[3:].view(-1, 1)

    # compute the rotation matrix using scaled rotation axis
    theta = torch.linalg.norm(rot_vec); u = rot_vec / theta
    if theta < 1e-10:  rot_mat = torch.eye(3)
    else:
        U = torch.Tensor([[0, -u[2], u[1]],
                          [u[2], 0, -u[0]],
                          [-u[1], u[0], 0]]).to(device)
        rot_mat = torch.eye(3).to(device) + torch.sin(theta) * U + (1 - torch.cos(theta)) * torch.matmul(U, U)

    # construct the post matrix from rotation matrix and translation vector
    pose_mat = torch.cat((rot_mat, trans_vec), dim = 1)
    return pose_mat


def synthesize(src_img, proj_pixels):
    # source image dimension should be (N, C, H, W)
    # projected pixel dimensioon should be (N, H, W, 2)
    if src_img.dim( ) < 4: src_img = src_img.unsqueeze(dim = 0)
    if proj_pixels.dim( ) < 4: proj_pixels = proj_pixels.unsqueeze(dim = 0)

    # synthesize some target image using source image and pixels from the target image project to source image
    return functional.grid_sample(input = src_img, grid = proj_pixels, padding_mode = "border", align_corners = True)


def synthesize_from_depths(intrinsic_mat, intrinsic_inv, pose_mat, depths, width, height, src_img, device):
    synthesized_imgs = list( )
    depths = arange(0, 80, 80 / depths)

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
