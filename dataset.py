import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf

from PIL import Image
from pathlib import Path
from collections import defaultdict


class TrainingData(Dataset):
    def __init__(self, seq_len: int, device: str = "cpu", train_root: str | Path = "kitti_data"):
        self.device = device
        self.seq_len = seq_len

        # convert dataset root to path object
        train_root = Path(train_root)

        # read filepaths from dataset_splits/train_files.txt
        with open("dataset_splits/train_files.txt") as train_files:
            contents = train_files.read( )
            list_contents = contents.split("\n")

            # group files according to directory
            grouped_content = defaultdict(list)
            for content in list_contents:
                train_dir, file_name = content.split(" ")
                if not (train_root / train_dir / (file_name + ".png")).is_file( ): continue
                grouped_content[train_dir].append(train_root / train_dir / (file_name + ".png"))
        
        image_paths = list( )
        for files in grouped_content.values( ):
            for idx in range(len(files) - seq_len + 1):
                image_paths.append(files[idx: idx + seq_len])
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
    def __init__(self, seq_len: int, device: str = "cpu", test_root: str | Path = "kitti_data", depth_root: str | Path = "data_depth_annotated"):
        self.device = device
        self.seq_len = seq_len

        # convert dataset roots to path objects
        test_root = Path(test_root)
        depth_root = Path(depth_root)

        # read filepaths from dataset_splits/test_files.txt
        with open("dataset_splits/test_files.txt") as test_files:
            # read filepaths from dataset_splits/depth_files.txt
            with open("dataset_splits/depth_files.txt") as depth_files:
                test_contents = test_files.read( )
                depth_contents = depth_files.read( )

                test_list_contents = test_contents.split("\n")
                depth_list_contents = depth_contents.split("\n")

                # group files according to directory
                grouped_img_content = defaultdict(list)
                grouped_depth_content = defaultdict(list)
                for test_content, depth_content in zip(test_list_contents, depth_list_contents):
                    test_dir, test_file_name = test_content.split(" ")
                    depth_dir, depth_file_name = depth_content.split(" ")

                    if not (test_root / test_dir / (test_file_name + ".png")).is_file( ): continue
                    if not (depth_root / depth_dir / (depth_file_name + ".png")).is_file( ): continue

                    grouped_img_content[test_dir].append(test_root / test_dir / (test_file_name + ".png"))
                    grouped_depth_content[depth_dir].append(depth_root / depth_dir / (depth_file_name + ".png"))
        
        image_paths = list( ); depth_paths = list( )
        for img_files, depth_files in zip(grouped_img_content.values( ), grouped_depth_content.values( )):
            for idx in range(len(img_files) - seq_len + 1):
                image_paths.append(img_files[idx: idx + seq_len])
            for idx in range(len(depth_files) - seq_len + 1):
                depth_paths.append(depth_files[idx + seq_len - 1])
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
