import torch
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
from pathlib import Path


class TrainingData(Dataset):
    def __init__(self, seq_len: int, device, transform = None, image_path = "kitti_data") -> None:
        self.device = device
        self.transform = transform

        # convert image path to path object
        image_path = Path(image_path)

        # training data roots
        train_data_roots = ["2011_09_26", "2011_09_28",
                            "2011_09_29", "2011_09_30"]

        # get subdirectories of training data roots
        train_data_subdirectories = { }
        for train_data_root in train_data_roots:
            if not (image_path / train_data_root).is_dir( ): continue

            train_data_subdirectories[train_data_root] = list( )
            for subdirectory in (image_path / train_data_root).iterdir( ):
                if subdirectory.is_dir( ): train_data_subdirectories[train_data_root].append(subdirectory)

        # get file paths of training data images
        train_left = { }
        for subdirectories in train_data_subdirectories.values( ):
            for subdirectory in subdirectories:
                train_left[subdirectory] = list( )
                for left_img in (subdirectory / "image_02/data").iterdir( ):
                    if left_img.suffix != ".png": continue
                    train_left[subdirectory].append(left_img)

        # concatenate adjacent images to sequences of specified length
        seq_train_left = [ ]
        for subdirectories in train_data_subdirectories.values( ):
            for subdirectory in subdirectories:
                for idx in range(len(train_left[subdirectory]) - seq_len + 1):
                    seq_train_left.append(train_left[subdirectory][idx:idx + seq_len])
        self.train_left = seq_train_left

        # printing training data statistics
        print("\nTraining Data Directories...\n")
        for subdirectories in train_data_subdirectories.values( ):
            count = 0
            for subdirectory in subdirectories:
                count += 1

                if count < 8: print(subdirectory)
                elif count == 8:
                    print(str(subdirectory) + "..."); break
            print("\n")

    def __len__(self) -> int:
        return len(self.train_left)


    def __getitem__(self, index: int) -> torch.Tensor:
        # returns a sequence of images indexed by index
        img_seq = list( )
        to_tensor = transforms.Compose([transforms.ToTensor( )])
        for img_path in self.train_left[index]:
            img = Image.open(img_path)
            if self.transform: img = self.transform(img)
            if not torch.is_tensor(img): img = to_tensor(img)
            img_seq.append(img)
        
        img_seq = torch.stack(img_seq)
        img_seq = img_seq.to(self.device)    
        return img_seq


class TestingData(Dataset):
    def __init__(self, seq_len: int, device, transform = None, image_path = "kitti_data", depth_path = "data_depth_annotated") -> None:
        self.device = device
        self.transform = transform
    
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
                for left_img, depth in zip((image_subdirectory / "image_02/data").iterdir( ), (depth_subdirectory / "proj_depth/groundtruth/image_02").iterdir( )):
                    if left_img.suffix != ".png" or depth.suffix != ".png": continue
                    test_left[image_subdirectory].append(left_img)
                    test_depth[depth_subdirectory].append(depth)

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
        to_tensor = transforms.Compose([transforms.ToTensor( )])
        for img_path in self.test_left[index]:
            img = Image.open(img_path)
            if self.transform: img = self.transform(img)
            if not torch.is_tensor(img): img = to_tensor(img)
            img_seq.append(img)
        img_seq = torch.stack(img_seq)
        img_seq = img_seq.to(self.device)

        # pair with corresponding depth
        depth = Image.open(self.test_depth[index])
        if self.transform: depth = self.transform(depth)
        if not torch.is_tensor(depth): depth = to_tensor(depth)
        depth = depth.to(self.device)

        return (img_seq, depth)


    def __len__(self) -> int:
        return len(self.test_left)
