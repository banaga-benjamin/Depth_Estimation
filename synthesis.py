import torch
import numpy as np
from torch.nn import functional


def construct_pose_batch(pose_vectors: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
        recovers a 3x4 pose matrix from a vector of size [6] \n
        input: pose vectors (N, 6) and device to store tensors \n
        output: pose matrices (N, 3, 4)
    """

    # obtain rotation and translation vectors
    N = pose_vectors.shape[0]
    rot_vecs = pose_vectors[:, :3]
    trans_vecs = pose_vectors[:, 3:]

    # obtain rotation angles and axes
    theta = torch.norm(rot_vecs, dim = 1, keepdim = True)
    u = rot_vecs / (theta + 1e-6)

    # batch-compute U matrices
    U = torch.zeros(N, 3, 3, device=device)
    U[:, 0, 1] = -u[:, 2]; U[:, 0, 2] = +u[:, 1]
    U[:, 1, 0] = +u[:, 2]; U[:, 1, 2] = -u[:, 0]
    U[:, 2, 0] = -u[:, 1]; U[:, 2, 1] = +u[:, 0]

    # compute rotation matrices
    eye = torch.eye(3, device = device).expand(N, 3, 3)
    rot_mats = eye + torch.sin(theta).view(N, 1, 1) * U + (1 - torch.cos(theta)).view(N, 1, 1) * torch.matmul(U, U)

    pose_mats = torch.cat([rot_mats, trans_vecs.unsqueeze(2)], dim = 2)
    return pose_mats


def synthesize(src_img: torch.Tensor, proj_pixels: torch.Tensor) -> torch.Tensor:
    """
        synthesizes a target image from a source image and pixels of target image projected to the source image \n
        input: source images (N, C, H, W) and pixel projection coordinates (N, H, W, 2) \n
        output: target image reconstructed from source image and pixel projections
    """

    # source image dimension should be (N, C, H, W)
    # projected pixel dimension should be (N, H, W, 2)
    if src_img.dim( ) < 4: src_img = src_img.unsqueeze(dim = 0)
    if proj_pixels.dim( ) < 4: proj_pixels = proj_pixels.unsqueeze(dim = 0)

    # normalize projection pixels
    H = src_img.size(dim = 2); W = src_img.size(dim = 3)
    proj_pixels[..., 0] = (proj_pixels[..., 0] / (W - 1)) * 2 - 1
    proj_pixels[..., 1] = (proj_pixels[..., 1] / (H - 1)) * 2 - 1

    # synthesize some target image using source image and pixels from the target image projected to source image
    return functional.grid_sample(input = src_img, grid = proj_pixels, padding_mode = "border", align_corners = True)


def synthesize_from_depths(intrinsic_mat: torch.Tensor, intrinsic_inv: torch.Tensor, pose_mat: torch.Tensor, depths: np.ndarray,
                           width: int, height: int, src_img: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
        synthesizes a target image from a source image, relative pose matrix, and a sequence of depths for initializing uniform depth maps \n
        input: camera intrinsic matrices (4, 4), pose matrix (3, 4), depth sequences [D], image width W and height H,
        source image (C, H, W), and device to store tensors \n
        output: synthesized target images (D, C, H, W)
    """

    synthesized_imgs = list( )
    for depth in depths: # for each depth in supplied depths
        # create a uniform depth map
        depth_map = torch.ones(height, width, device = device) * depth

        # project some target image to the source image
        proj_pixels = reproject_from_depth(intrinsic_mat, intrinsic_inv, pose_mat, depth_map, width, height, device)

        # synthesize the target image
        synthesized_img = synthesize(src_img, proj_pixels)

        # synthesized img should be of dimensions (C, H, W)
        if synthesized_img.dim( ) > 3: synthesized_img = torch.squeeze(synthesized_img)
        synthesized_imgs.append(synthesized_img)
    return torch.stack(synthesized_imgs)


def reproject_from_depth(intrinsic_mat: torch.Tensor, intrinsic_inv: torch.Tensor, pose_mat: torch.Tensor,
                         depth_map: torch.Tensor, width: int = 640, height: int = 192, device: str = "cpu") -> torch.Tensor:
    """
        projects the pixels of some target image to a source image using a relative pose matrix and a depth map \n
        input: camera intrinsic matrices (4, 4), pose matrix (3, 4), depth map (1, H, W), image width W and height H, and device to store tensors \n
        output: pixel coordinates of the pixels of the target image projected to the source image (H, W, 2)
    """

    # create 3D pixel grid (H, W, 3)
    x_coords, y_coords = torch.meshgrid(
        torch.arange(width, dtype = torch.float32, device = device),
        torch.arange(height, dtype = torch.float32, device = device),
        indexing = 'xy'
    )
    pixel_coords = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim = -1) # (H, W, 3)

    # back-project to camera space (H, W, 3)
    camera_coords = torch.matmul(pixel_coords, intrinsic_inv[:3, :3].T)

    # scale by depth (handle depth map dimensions)
    if depth_map.dim() == 4: depth_map = depth_map.squeeze(1)
    depth_map = depth_map.squeeze( ).unsqueeze(-1)                                          # (H, W, 1)
    points_3d = camera_coords * depth_map                                                   # (H, W, 3)

    # transform to source frame (add homogeneous coordinate for pose)
    points_3d_hom = torch.cat([points_3d, torch.ones_like(points_3d[..., :1])], dim = -1)   # (H, W, 4)
    points_3d_transformed = torch.matmul(points_3d_hom, pose_mat.T)                         # (H, W, 3)

    # project to 2D
    projected = torch.matmul(points_3d_transformed, intrinsic_mat[:3, :3].T)
    z = torch.where(torch.abs(projected[..., 2]) < 1e-6, torch.ones_like(projected[..., 2]) * 1e-6, projected[..., 2])
    output_pixels = torch.stack([projected[..., 0] / z, projected[..., 1] / z], dim = -1)   # (H, W, 2)

    return output_pixels


def depth_from_costs(cost_volumes: torch.Tensor, num_channels: int = 80, sid: bool = True, device: str = "cpu") -> torch.Tensor:
    """
        determines a possible depth map from a cost volume \n
        input: cost volumes (N, C, H, W), number of channels C, flag to use SID, device to store tensors \n
        output: possible depth determined from cost volume
    """

    # compute softmin weights (smaller cost → higher weight)
    weights = functional.softmin(torch.square(torch.exp(cost_volumes)) - 1, dim = 1)
    
    # compute channels [0, 1, ..., C - 1] and reshape for broadcasting
    channels = torch.arange(num_channels, device = device).float( )
    channels = channels.view(1, -1, 1, 1)
    
    # compute weighted average of channels (soft argmin)
    min_channels = (weights * channels).sum(dim = 1, keepdim = True)

    # map channels to corresponding depth
    if sid: return torch.exp((min_channels + 1) * torch.log(torch.tensor(2.0)) / num_channels) - 1
    else: return (min_channels / num_channels) + (1 / num_channels)