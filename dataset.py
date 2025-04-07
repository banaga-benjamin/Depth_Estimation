import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf

from PIL import Image
from pathlib import Path


class TrainingData(Dataset):
    def __init__(self, seq_len: int, device: str = "cpu", image_path: str | Path = "kitti_data"):
        self.device = device
        self.seq_len = seq_len

        # convert image path to path object
        image_path = Path(image_path)

        # training data roots
        data_roots = ["2011_09_26", "2011_09_28", "2011_09_29", "2011_09_30"]

        # get subdirectories of training data roots
        data_subdirectories = { }
        for data_root in data_roots:
            if not (image_path / data_root).is_dir( ): continue

            data_subdirectories[data_root] = list( )
            for subdirectory in (image_path / data_root).iterdir( ):
                if subdirectory.is_dir( ): data_subdirectories[data_root].append(subdirectory)

        # get file paths of training data images
        temp_image_paths = { }
        for subdirectories in data_subdirectories.values( ):
            for subdirectory in subdirectories:
                temp_image_paths[subdirectory] = list( )
                for img in (subdirectory / "image_02/data").iterdir( ):
                    if img.suffix == ".png": temp_image_paths[subdirectory].append(img)

        # collect adjacent image paths as specified by seq len
        image_paths = list( )
        for image_path_list in temp_image_paths.values( ):
            for idx in range(0, len(image_path_list) - seq_len + 1):
                image_paths.append(image_path_list[idx: idx + seq_len])
        self.image_paths = image_paths


    def __getitem__(self, index: int) -> torch.Tensor:
        # determine augmentations
        flip = torch.rand(1) < 0.5
        angle = (torch.rand(1) * 30 - 15).item( )

        # returns a sequence of images indexed by index
        img_seq = list( )
        for img_path in self.image_paths[index]:
            img = Image.open(img_path)

            # flip and rotate
            if flip: img = tf.hflip(img)
            img = tf.rotate(img, angle)

            img = tf.to_tensor(tf.resize(img, size = (192, 640)))
            img_seq.append(img)
        
        img_seq = torch.stack(img_seq)
        img_seq = img_seq.to(self.device)
        return img_seq


    def __len__(self) -> int:
        return len(self.image_paths)


class TestingData(Dataset):
    def __init__(self, seq_len: int, device: str = "cpu", image_path: str | Path = "kitti_data", depth_path: str | Path = "data_depth_annotated"):
        self.device = device
        self.seq_len = seq_len
    
        # set training data subdirectories
        image_path = Path(image_path)
        depth_path = Path(depth_path)

        # testing data roots
        data_roots = ["2011_10_03"]

        # get subdirectories of testing data roots
        image_subdirectories = { }
        depth_subdirectories = { }
        for data_root in data_roots:
            if not (image_path / data_root).is_dir( ): continue
            if not (depth_path / data_root).is_dir( ): continue
            
            image_subdirectories[data_root] = list( )
            depth_subdirectories[data_root] = list( )
            for image_subdirectory, depth_subdirectory in zip((image_path / data_root).iterdir( ), (depth_path / data_root).iterdir( )):
                if image_subdirectory.is_dir( ) and depth_subdirectory.is_dir( ):
                    image_subdirectories[data_root].append(image_subdirectory)
                    depth_subdirectories[data_root].append(depth_subdirectory)

        # get file paths of testing data images and depths
        temp_image_paths = { }
        temp_depth_paths = { }
        for image_subs, depth_subs in zip(image_subdirectories.values( ), depth_subdirectories.values( )):
            for image_sub, depth_sub in zip(image_subs, depth_subs):
                temp_image_paths[image_sub] = list( )
                temp_depth_paths[depth_sub] = list( )

                for img in (image_sub / "image_02/data").iterdir( ):
                    if img.suffix != ".png": continue
                    temp_image_paths[image_sub].append(img)
                for depth in (depth_sub / "proj_depth/groundtruth/image_02").iterdir( ):
                    if depth.suffix != ".png": continue
                    temp_depth_paths[depth_sub].append(depth)

                # for alignment
                temp_image_paths[image_sub] = temp_image_paths[image_sub][5:-5]
        
        image_paths = list( )
        depth_paths = list( )
        for image_path_list, depth_path_list in zip(temp_image_paths.values( ), temp_depth_paths.values( )):
            for idx in range(0, len(image_path_list) - seq_len + 1):
                image_paths.append(image_path_list[idx: idx + seq_len])
            for idx in range(0, len(depth_path_list) - seq_len + 1):
                depth_paths.append(depth_path_list[idx + seq_len - 1])
        self.image_paths = image_paths
        self.depth_paths = depth_paths


    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        # collect sequence of images indexed by index
        img_seq = list( )
        for img_path in self.image_paths[index]:
            img = Image.open(img_path)
            img = tf.to_tensor(tf.resize(img, size = (192, 640)))
            img_seq.append(img)
        img_seq = torch.stack(img_seq)
        img_seq = img_seq.to(self.device)

        # pair with corresponding depth
        depth = Image.open(self.depth_paths[index])
        depth = tf.to_tensor(tf.resize(depth, size = (192, 640)))

        # normalize depth map to [0, 1] and set to device
        depth = (depth / (2 ** 16)).to(self.device)
        return (img_seq, depth)


    def __len__(self) -> int:
        return len(self.image_paths)
