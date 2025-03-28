import torch
from torch.nn import functional


def cost_volumes(target_img, synthesized_imgs):
    # target image should be of dimension (C, H, W)
    if target_img.dim( ) > 3: target_img = target_img.squeeze( )
    target_imgs = torch.stack([target_img] * len(synthesized_imgs[0]))
    target_imgs = torch.stack([target_imgs] * len(synthesized_imgs))

    # get the mean of the differences along dimension C
    return (target_imgs - synthesized_imgs).mean(dim = 2)


def reprojection_loss(preds, targets):
    # target and predicted images should be of dimensions (N, C, H, W)
    if targets.dim( ) < 4: targets = targets.unsqueeze(dim = 0)
    if preds.dim( ) < 4: preds = preds.unsqueeze(dim = 0)

    # apply padding to retain dimensions
    x = functional.pad(preds, pad = (1, 1, 1, 1)); y = functional.pad(targets, pad = (1, 1, 1, 1))

    # compute the SSIM loss
    mu_x = functional.avg_pool2d(x, kernel_size = 3, stride = 1)
    mu_y = functional.avg_pool2d(y, kernel_size = 3, stride = 1)

    sigma_x  = functional.avg_pool2d(x ** 2, kernel_size = 3, stride = 1) - mu_x ** 2
    sigma_y  = functional.avg_pool2d(y ** 2, kernel_size = 3, stride = 1) - mu_y ** 2
    sigma_xy = functional.avg_pool2d(x * y, kernel_size = 3, stride = 1) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + (0.01 ** 2)) * (2 * sigma_xy + (0.03 ** 2))
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + (0.01 ** 2)) * (sigma_x + sigma_y + (0.03 ** 2))

    ssim_loss = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1).mean(dim = 1, keepdim = True)

    # compute L1 loss
    l1_loss = torch.abs(targets - preds).mean(dim = 1, keepdim = True)

    # return the mean of the reprojection error
    return (0.05 * ssim_loss + 0.15 * l1_loss).mean( )


def regularization_term(depths, target_imgs):
    # get the gradients along the x and y axes
    depth_x, depth_y = torch.gradient(depths, dim = (-2, -1))
    target_x, target_y = torch.gradient(target_imgs, dim = (-2, -1))

    # get the absolute values of the means of the gradients
    depth_x = depth_x.mean( ).abs( ); depth_y = depth_y.mean( ).abs( )
    target_x = target_x.mean( ).abs( ); target_y = target_y.mean( ).abs( )
    return depth_x * torch.exp(-target_x) + depth_y * torch.exp(-target_y)


def rmse(pred_depths, depths):
    # returns root of the mean squared error
    return torch.sqrt(functional.mse_loss(pred_depths, depths))


def rmsle(pred_depths, depths, eps = 1e-6):
    # returns root of the mean squared error of log values
    return rmse(torch.log(pred_depths + eps), torch.log(depths + eps))


def abs_rel(pred_depths, depths, eps = 1e-6):
    # returns the mean of the absolute relative error
    return torch.mean(torch.abs(pred_depths - depths) / (depths + eps))


def sq_rel(pred_depths, depths, eps = 1e-6):
    # returns the mean of the squared relative error
    return torch.mean(((pred_depths - depths) ** 2) / (depths + eps))
