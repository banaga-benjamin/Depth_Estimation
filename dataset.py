import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf

from PIL import Image
from pathlib import Path


class TrainingData(Dataset):
    def __init__(self, seq_len: int, device, image_path = "kitti_data") -> None:
        self.device = device
        self.seq_len = seq_len

        # convert image path to path object
        image_path = Path(image_path)

        # training data roots
        data_roots = ["2011_09_26", "2011_09_28",
                            "2011_09_29", "2011_09_30"]

        # get subdirectories of training data roots
        data_subdirectories = { }
        for data_root in data_roots:
            if not (image_path / data_root).is_dir( ): continue

            data_subdirectories[data_root] = list( )
            for subdirectory in (image_path / data_root).iterdir( ):
                if subdirectory.is_dir( ): data_subdirectories[data_root].append(subdirectory)

        # get file paths of training data images
        image_paths = [ ]
        for subdirectories in data_subdirectories.values( ):
            for subdirectory in subdirectories:
                for left_img in (subdirectory / "image_02/data").iterdir( ):
                    if left_img.suffix == ".png": image_paths.append(left_img)
        self.image_paths = image_paths

        # printing training data statistics
        print("\nTraining Data Directories...\n")
        for subdirectories in data_subdirectories.values( ):
            count = 0
            for subdirectory in subdirectories:
                count += 1

                if count < 8: print(subdirectory)
                elif count == 8:
                    print(str(subdirectory) + "..."); break
            print("\n")

    def __len__(self) -> int:
        return len(self.image_paths) - self.seq_len + 1


    def __getitem__(self, index: int) -> torch.Tensor:
        # determine augmentations
        flip = torch.rand(1) < 0.5
        angle = (torch.rand(1) * 30 - 15).item( )

        # returns a sequence of images indexed by index
        img_seq = list( )
        for img_path in self.image_paths[index: index + self.seq_len]:
            img = Image.open(img_path)

            # flip and rotate
            if flip: img = tf.hflip(img)
            img = tf.rotate(img, angle)

            img = tf.to_tensor(tf.resize(img, size = (192, 640)))
            img_seq.append(img)
        
        img_seq = torch.stack(img_seq)
        img_seq = img_seq.to(self.device)
        return img_seq


class TestingData(Dataset):
    def __init__(self, seq_len: int, device, image_path = "kitti_data", depth_path = "data_depth_annotated") -> None:
        self.device = device
    
        # set training data subdirectory
        image_path = Path(image_path)
        depth_path = Path(depth_path)

        # testing data roots
        test_data_roots = ["2011_10_03"]

        # get subdirectories of testing data roots
        test_image_subdirectories = { }
        test_depth_subdirectories = { }
        for test_data_root in test_data_roots:
            if not (image_path / test_data_root).is_dir( ): continue
            if not (depth_path / test_data_root).is_dir( ): continue
            
            test_image_subdirectories[test_data_root] = list( )
            test_depth_subdirectories[test_data_root] = list( )
            for image_subdirectory, depth_subdirectory in zip((image_path / test_data_root).iterdir( ), (depth_path / test_data_root).iterdir( )):
                if image_subdirectory.is_dir( ) and depth_subdirectory.is_dir( ):
                    test_image_subdirectories[test_data_root].append(image_subdirectory)
                    test_depth_subdirectories[test_data_root].append(depth_subdirectory)

        # get file paths of testing data images and depths
        test_left = { }
        test_depth = { }
        for image_subdirectories, depth_subdirectories in zip(test_image_subdirectories.values( ), test_depth_subdirectories.values( )):
            for image_subdirectory, depth_subdirectory in zip(image_subdirectories, depth_subdirectories):
                test_left[image_subdirectory] = list( )
                test_depth[depth_subdirectory] = list( )

                for left_img in (image_subdirectory / "image_02/data").iterdir( ):
                    if left_img.suffix != ".png": continue
                    test_left[image_subdirectory].append(left_img)
                for depth in (depth_subdirectory / "proj_depth/groundtruth/image_02").iterdir( ):
                    if depth.suffix != ".png": continue
                    test_depth[depth_subdirectory].append(depth)

                # for alignment
                test_left[image_subdirectory] = test_left[image_subdirectory][5:-5]

        # concatenate adjacent images to sequences of specified length and match with corresponding depths
        seq_test_left = [ ]
        seq_test_depth = [ ]
        for img_subdirectories, depth_subdirectories in zip(test_image_subdirectories.values( ), test_depth_subdirectories.values( )):
            for img_subdirectory, depth_subdirectory in zip(img_subdirectories, depth_subdirectories):
                for idx in range(len(test_left[img_subdirectory]) - seq_len + 1):
                    seq_test_left.append(test_left[img_subdirectory][idx:idx + seq_len])
                    seq_test_depth.append(test_depth[depth_subdirectory][idx + seq_len - 1])
        
        self.seq_len = seq_len
        self.test_left = seq_test_left
        self.test_depth = seq_test_depth

        # printing testing data statistics
        print("\nTest Data Directories...\n")
        for img_subdirectories, depth_subdirectories in zip(test_image_subdirectories.values( ), test_depth_subdirectories.values( )):
            count = 0
            for img_subdirectory, depth_subdirectory in zip(img_subdirectories, depth_subdirectories):
                count += 1

                if count <= 8: print(img_subdirectory, "\t", depth_subdirectory)
                elif count == 8:
                    print(str(img_subdirectory, depth_subdirectory) + "..."); break
            print("\n")

    def __getitem__(self, index: int) -> torch.Tensor:
        # collect sequence of images indexed by index
        img_seq = list( )
        for img_path in self.test_left[index]:
            img = Image.open(img_path)
            img = tf.to_tensor(tf.resize(img, size = (192, 640)))
            img_seq.append(img)
        img_seq = torch.stack(img_seq)
        img_seq = img_seq.to(self.device)

        # pair with corresponding depth
        depth = Image.open(self.test_depth[index])
        depth = tf.to_tensor(tf.resize(depth, size = (192, 640)))

        # normalize depth map to [0, 1] and set to device
        depth = (depth / (2 ** 16)).to(self.device)
        return (img_seq, depth)


    def __len__(self) -> int:
        return len(self.test_left)
