import torch
from torch.nn import functional


def cost_volumes(target_img, synthesized_imgs):
    # target image should be of dimension (C, H, W)
    if target_img.dim( ) > 3: target_img = target_img.squeeze( )
    target_imgs = torch.stack([target_img] * len(synthesized_imgs[0]))
    target_imgs = torch.stack([target_imgs] * len(synthesized_imgs))

    # get the mean of the differences along dimension C
    return (target_imgs - synthesized_imgs).mean(dim = 2)


def reprojection_error(preds, targets):
    # target and predicted images should be of dimensions (N, C, H, W)
    if targets.dim( ) < 4: targets = targets.unsqueeze(dim = 0)
    if preds.dim( ) < 4: preds = preds.unsqueeze(dim = 0)

    # apply padding to retain dimensions
    x = functional.pad(preds, pad = (1, 1, 1, 1), mode = 'reflect'); y = functional.pad(targets, pad = (1, 1, 1, 1), mode = 'reflect')

    # compute the SSIM loss
    mu_x = functional.avg_pool2d(x, kernel_size = 3, stride = 1)
    mu_y = functional.avg_pool2d(y, kernel_size = 3, stride = 1)

    sigma_x  = functional.avg_pool2d(x ** 2, kernel_size = 3, stride = 1) - mu_x ** 2
    sigma_y  = functional.avg_pool2d(y ** 2, kernel_size = 3, stride = 1) - mu_y ** 2
    sigma_xy = functional.avg_pool2d(x * y, kernel_size = 3, stride = 1) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + (0.01 ** 2)) * (2 * sigma_xy + (0.03 ** 2))
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + (0.01 ** 2)) * (sigma_x + sigma_y + (0.03 ** 2))

    ssim_loss = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
    return ssim_loss


def reprojection_loss(preds, targets):
    # compute L1 losses and SSIM losses
    l1_losses = torch.abs(targets - preds)
    ssim_losses = reprojection_error(preds, targets)
    reprojection_losses = (0.85 * ssim_losses) + (0.15 * l1_losses)
    return reprojection_losses.min(dim = 0, keepdim = True)[0]


def regularization_term(depths, target_imgs):
    # get the gradients along the x and y axes
    depth_x, depth_y = torch.gradient(depths, dim = (-2, -1))
    target_x, target_y = torch.gradient(target_imgs, dim = (-2, -1))

    # get the absolute values of the gradients
    depth_x = depth_x.abs( ); depth_y = depth_y.abs( )
    target_x = target_x.abs( ); target_y = target_y.abs( )

    # broadcast depth gradients to channels of target image
    depth_x = torch.cat([depth_x] * target_x.size(dim = 1), dim = 1)
    depth_y = torch.cat([depth_y] * target_y.size(dim = 1), dim = 1)
    
    return torch.mean(depth_x / torch.exp(target_x) + depth_y / torch.exp(target_y), dim = 0, keepdim = True)


def depth_regularization(depths):
    # penalty for values very close to zero and values very close to one
    depth_mean = torch.mean(torch.mean(depths, dim = (1, 2, 3)))
    return (1 / torch.exp(12 * depth_mean)) + (1 / torch.exp(1 - depth_mean))


def rmse(pred_depths, depths):
    # calculate errors only for values for which depths > 0
    errors = torch.where(depths > 0, torch.square(pred_depths - depths), torch.zeros_like(depths))
    
    # return root of the mean squared error of values
    return torch.mean(torch.sqrt(torch.mean(errors, dim = (1, 2, 3))))


def rmsle(pred_depths, depths, eps = 1e-6):
    # calculate errors only for values for which depths > 0
    errors = torch.where(depths > 0, torch.square(torch.log(pred_depths + eps) - torch.log(depths + eps)), torch.zeros_like(depths))
    
    # return root of the mean squared error of values
    return torch.mean(torch.sqrt(torch.mean(errors, dim = (1, 2, 3))))


def abs_rel(pred_depths, depths):
    # calculate errors only for values for which depths > 0
    errors = torch.where(depths > 0, torch.abs(pred_depths - depths) / (depths), torch.zeros_like(depths))

    # return squared relative errors
    return torch.mean(torch.mean(errors, dim = (1, 2, 3)))


def sq_rel(pred_depths, depths):
    # calculate errors only for values for which depths > 0
    errors = torch.where(depths > 0, torch.square(pred_depths - depths) / (depths), torch.zeros_like(depths))

    # return squared relative errors
    return torch.mean(torch.mean(errors, dim = (1, 2, 3)))
