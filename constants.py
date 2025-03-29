import torch
import numpy as np
from os import cpu_count


# constants for image dimensions
HEIGHT = 192; WIDTH = 640
COST_HEIGHT = 96; COST_WIDTH = 320

# constants for cost volume depths
USE_SID = False
COST_DEPTHS = 64
UID_DEPTHS = np.arange(0.0, 1.0, 1 / COST_DEPTHS)
SID_DEPTHS = np.array([np.exp(x * np.log(2) / COST_DEPTHS) - 1 for x in range(COST_DEPTHS)])

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

# constants related to training
EPOCHS = 1
SEQ_LEN = 4
BATCH_SIZE = 4
NUM_RANDOM_TRANS = 16 - 1
NUM_WORKERS = cpu_count( )
