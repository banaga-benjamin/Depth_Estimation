import torch
from torch.nn import functional


def cost_volumes(target_img: torch.Tensor, synthesized_imgs: torch.Tensor) -> torch.Tensor:
    """
        calculates the cost volumes between a target image and synthesized images \n
        input: a target image (C, H, W) and synthesized images (N, C, H, W) \n
        output: cost volume between the target image and the synthesized images
    """

    # target image should be of dimension (C, H, W)
    # synthesized images are of dimension (N, C, H, W)
    if target_img.dim( ) > 3: target_img = target_img.squeeze( )
    target_imgs = torch.stack([target_img] * synthesized_imgs.size(dim = 0), dim = 0)

    # get the mean along dimension C of the absolute value of the differences
    return torch.abs(target_imgs - synthesized_imgs).mean(dim = 1)


def reprojection_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
        calculates the reprojection error between predicted and target images \n
        input: predicted images (N, C, H, W) and target images (N, C, H, W) \n
        output: reprojection error between predicted and target images
    """

    # target and predicted images should be of dimensions (N, C, H, W)
    if targets.dim( ) < 4: targets = targets.unsqueeze(dim = 0)
    if preds.dim( ) < 4: preds = preds.unsqueeze(dim = 0)

    # apply padding to retain dimensions
    x = functional.pad(preds, pad = (1, 1, 1, 1), mode = 'reflect')
    y = functional.pad(targets, pad = (1, 1, 1, 1), mode = 'reflect')

    # compute the means
    mu_x = functional.avg_pool2d(x, kernel_size = 3, stride = 1)
    mu_y = functional.avg_pool2d(y, kernel_size = 3, stride = 1)

    # compute the variances and covariance
    sigma_x  = functional.avg_pool2d(x ** 2, kernel_size = 3, stride = 1) - mu_x ** 2
    sigma_y  = functional.avg_pool2d(y ** 2, kernel_size = 3, stride = 1) - mu_y ** 2
    sigma_xy = functional.avg_pool2d(x * y, kernel_size = 3, stride = 1) - mu_x * mu_y

    # compute the numerator and denominator of SSIM
    SSIM_n = (2 * mu_x * mu_y + (0.01 ** 2)) * (2 * sigma_xy + (0.03 ** 2))
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + (0.01 ** 2)) * (sigma_x + sigma_y + (0.03 ** 2))

    # compute weighted SSIM, clamped to [0, 1]
    ssim_loss = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
    return ssim_loss


def reprojection_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
        calculates the reprojection loss between predicted and target images \n
        input: predicted images (N, C, H, W) and target images (N, C, H, W) \n
        output: reprojection loss between predicted and target images
    """

    # compute L1 losses and SSIM losses
    l1_losses = torch.abs(targets - preds)
    ssim_losses = reprojection_error(preds, targets)

    # combine L1 losses and SSIM losses with weights
    reprojection_losses = (0.85 * ssim_losses) + (0.15 * l1_losses)

    # get mean along channel dimension
    reprojection_losses = torch.mean(reprojection_losses, dim = 1, keepdim = True)

    # get the minimums along batch dimension
    return reprojection_losses.min(dim = 0, keepdim = True)[0]


def regularization_term(depths: torch.Tensor, target_imgs: torch.Tensor) -> torch.Tensor:
    """
        calculates the depth regularization term w.r.t. to depth maps and corresponding target images \n
        input: depth maps (N, 1, H, W) and target images (N, C, H, W) \n
        output: depth regularization term w.r.t. depth maps and target images
    """

    # get the gradients along the x and y axes
    depth_x, depth_y = torch.gradient(depths, dim = (-1, -2))
    target_x, target_y = torch.gradient(target_imgs, dim = (-1, -2))

    # get the absolute values of the gradients
    depth_x = depth_x.abs( ); depth_y = depth_y.abs( )
    target_x = target_x.abs( ); target_y = target_y.abs( )

    # broadcast depth gradients to channels of target image
    depth_x = torch.cat([depth_x] * target_x.size(dim = 1), dim = 1)
    depth_y = torch.cat([depth_y] * target_y.size(dim = 1), dim = 1)

    # get mean of regularization term along channel dimension
    regularization_term = torch.mean(depth_x / torch.exp(target_x) + depth_y / torch.exp(target_y), dim = 1)
    
    return torch.mean(regularization_term)


def rmse(pred_depths, depths: torch.Tensor) -> torch.Tensor:
    """
        computes the RMSE between predicted and ground truth depths \n
        input: predicted depth (N, 1, H, W) and ground truth depths (N, 1, H, W) \n
        output: RMSE between predicted and ground truth depths
    """

    # calculate errors only for values for which ground truth depths > 0
    errors = torch.where(depths > 0, torch.square(pred_depths - depths), torch.zeros_like(depths))
    
    # return root of the mean squared error of values
    return torch.mean(torch.sqrt(torch.mean(errors, dim = (1, 2, 3))))


def rmsle(pred_depths: torch.Tensor, depths: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
        computes the RMSLE between predicted and ground truth depths \n
        input: predicted depth (N, 1, H, W) and ground truth depths (N, 1, H, W) \n
        output: RMSLE between predicted and ground truth depths
    """

    # calculate errors only for values for which depths > 0
    errors = torch.where(depths > 0, torch.square(torch.log(pred_depths + eps) - torch.log(depths + eps)), torch.zeros_like(depths))
    
    # return root of the mean squared error of log values
    return torch.mean(torch.sqrt(torch.mean(errors, dim = (1, 2, 3))))


def abs_rel(pred_depths: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    """
        computes the absolute relative error between predicted and ground truth depths \n
        input: predicted depth (N, 1, H, W) and ground truth depths (N, 1, H, W) \n
        output: absolute relative error between predicted and ground truth depths
    """

    # calculate errors only for values for which depths > 0
    errors = torch.where(depths > 0, torch.abs(pred_depths - depths) / (depths), torch.zeros_like(depths))

    # return absolute relative errors
    return torch.mean(torch.mean(errors, dim = (1, 2, 3)))


def sq_rel(pred_depths: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    """
        computes the squared relative error between predicted and ground truth depths \n
        input: predicted depth (N, 1, H, W) and ground truth depths (N, 1, H, W) \n
        output: squared relative error between predicted and ground truth depths
    """

    # calculate errors only for values for which depths > 0
    errors = torch.where(depths > 0, torch.square(pred_depths - depths) / (depths), torch.zeros_like(depths))

    # return squared relative errors
    return torch.mean(torch.mean(errors, dim = (1, 2, 3)))
