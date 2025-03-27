from torch import stack
from torch import Tensor
from torch import is_tensor
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
        for train_data_root in train_data_roots:
            for subdirectory in train_data_subdirectories[train_data_root]:
                train_left[subdirectory] = list( )
                for left_img in (subdirectory / "image_02/data").iterdir( ):
                    if left_img.suffix != ".png": continue
                    train_left[subdirectory].append(left_img)

        # concatenate adjacent images to sequences of specified length
        seq_train_left = [ ]
        for train_data_root in train_data_roots:
            for subdirectory in train_data_subdirectories[train_data_root]:
                for idx in range(len(train_left[subdirectory]) - seq_len + 1):
                    seq_train_left.append(train_left[subdirectory][idx:idx + seq_len])
        self.train_left = seq_train_left

        # printing training data statistics
        print("\nTraining Data Directories...\n")
        for train_data_root in train_data_roots:
            print("Training Data Root:", train_data_root)
            for subdirectory in train_data_subdirectories[train_data_root]: print(subdirectory)
            print("-" * 100, "\n")


    def __len__(self) -> int:
        return len(self.train_left)


    def __getitem__(self, index: int) -> Tensor:
        # returns a sequence of images indexed by index
        img_seq = list( )
        to_tensor = transforms.Compose([transforms.ToTensor( )])
        for img_path in self.train_left[index]:
            img = Image.open(img_path)
            if self.transform: img = self.transform(img)
            if not is_tensor(img): img = to_tensor(img)
            img_seq.append(img)
        
        img_seq = stack(img_seq)
        img_seq = img_seq.to(self.device)    
        return img_seq


class TestingData(Dataset):
    def __init__(self, seq_len: int, device, transform = None, image_path = "kitti_data") -> None:
        self.device = device
        self.transform = transform
    
        # set training data subdirectory
        image_path = Path(image_path)

        # testing data roots
        test_data_roots = ["2011_10_03"]

        # get subdirectories of testing data roots
        test_data_subdirectory = { }
        for test_data_root in test_data_roots:
            if not (image_path / test_data_root).is_dir( ): continue

            test_data_subdirectory[test_data_root] = list( )
            for subdirectory in (image_path / test_data_root).iterdir( ):
                if subdirectory.is_dir( ): test_data_subdirectory[test_data_root].append(subdirectory)

        # get file paths of testing data images
        test_left = { }
        for test_data_root in test_data_roots:
            for subdirectory in test_data_subdirectory[test_data_root]:
                test_left[subdirectory] = list( )
                for left_img in (subdirectory / "image_02/data").iterdir( ):
                    if left_img.suffix != ".png": continue
                    test_left[subdirectory].append(left_img)

        # concatenate adjacent images to sequences of specified length
        seq_test_left = [ ]
        for test_data_root in test_data_roots:
            for subdirectory in test_data_subdirectory[test_data_root]:
                for idx in range(len(test_left[subdirectory]) - seq_len + 1):
                    seq_test_left.append(test_left[subdirectory][idx:idx + seq_len])
        self.test_left = seq_test_left

        # printing testing data statistics
        print("\nTesting Data Directories...\n")
        for test_data_root in test_data_roots:
            print("Testing Data Root:", test_data_root)
            for subdirectory in test_data_subdirectory[test_data_root]: print(subdirectory)
            print("-" * 100, "\n")

    def __getitem__(self, index: int) -> Tensor:
        # returns a sequence of images indexed by index
        img_seq = list( )
        to_tensor = transforms.Compose([transforms.ToTensor( )])
        for img_path in self.test_left[index]:
            img = Image.open(img_path)
            if self.transform: img = self.transform(img)
            if not is_tensor(img): img = to_tensor(img)
            img_seq.append(img)
        img_seq = stack(img_seq)
        img_seq = img_seq.to(self.device)
        return img_seq


    def __len__(self) -> int:
        return len(self.test_left)
