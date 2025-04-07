import torch
import numpy as np


# training constants
EPOCHS = 1; SEQ_LEN = 4
BATCH_SIZE = 2; NUM_WORKERS = 0

# constants for image dimensions
MAX_DEPTH = 80
HEIGHT = 192; WIDTH = 640


# constants for cost volumes
COST_HEIGHT = 96; COST_WIDTH = 320

USE_SID = True; COST_DEPTHS = 80
UID_DEPTHS = (np.arange(0.0, 1.0, 1 / COST_DEPTHS) + (1 / COST_DEPTHS)) * MAX_DEPTH
SID_DEPTHS = np.array([np.exp((x + 1) * np.log(2) / COST_DEPTHS) - 1 for x in range(COST_DEPTHS)]) * MAX_DEPTH

DEPTHS = SID_DEPTHS if USE_SID else UID_DEPTHS


# intrinsic matrices
INTRINSIC_MAT = torch.Tensor(
            [[0.58, 0.00, 0.50, 0.00],
            [0.00, 1.92, 0.50, 0.00],
            [0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.00, 1.00]]
        )

# cost volume intrinsic matrices
COST_INTRINSIC_MAT = INTRINSIC_MAT.clone( )

COST_INTRINSIC_MAT[0, :] *= COST_WIDTH
COST_INTRINSIC_MAT[1, :] *= COST_HEIGHT
COST_INTRINSIC_INV = torch.linalg.pinv(COST_INTRINSIC_MAT)

# final image reprojection intrinsic matrices
IMG_INTRINSIC_MAT = INTRINSIC_MAT.clone( )

IMG_INTRINSIC_MAT[0, :] *= WIDTH
IMG_INTRINSIC_MAT[1, :] *= HEIGHT
IMG_INTRINSIC_INV = torch.linalg.pinv(IMG_INTRINSIC_MAT)
