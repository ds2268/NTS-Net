from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import itertools
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np


class AWEDataset(Dataset):
    """AWE (Annotated Web Ears) dataset"""

    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            train (bool): whether to use train or val dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.train_paths, self.train_y = self.prepare_train_anno()
        self.val_paths, self.val_y = self.prepare_val_anno()
        self.num_classes = len(set(self.train_y))
        self.classes = list(set(self.train_y))
        self.classes.sort()
        self.class_to_idx = {self.classes[idx]: idx for idx in range(len(self.classes))}

    def prepare_train_anno(self):
        train_path = self.root_dir + "/train/"
        train_subjects = sorted(filter(lambda f: not f.startswith('.'), os.listdir(train_path)))
        train_paths = []
        train_y = []
        for train_subject in train_subjects:
            subject_path = os.path.join(train_path, train_subject + "/")
            subject_samples = sorted(filter(lambda f: not f.startswith('.'), os.listdir(subject_path)))
            subject_train_paths = ["".join(item) for item in list(itertools.product([subject_path],
                                                                                    list(subject_samples)))]
            train_paths += subject_train_paths
            train_y += [int(train_subject)] * len(subject_samples)

        return train_paths, train_y

    def prepare_val_anno(self):
        val_path = self.root_dir + "/val/"
        val_subjects = sorted(filter(lambda f: not f.startswith('.'), os.listdir(val_path)))
        val_paths = []
        val_y = []
        for val_subject in val_subjects:
            subject_path = os.path.join(val_path, val_subject + "/")
            subject_samples = sorted(filter(lambda f: not f.startswith('.'), os.listdir(subject_path)))
            subject_val_paths = ["".join(item) for item in
                                 list(itertools.product([subject_path], list(subject_samples)))]
            val_paths += subject_val_paths
            val_y += [int(val_subject)] * len(subject_samples)

        return val_paths, val_y

    def __len__(self):
        if self.train:
            return len(self.train_paths)
        else:
            return len(self.val_paths)

    def __getitem__(self, idx):
        if self.train:
            img_path = self.train_paths[idx]
            image = imread(img_path, mode="RGB")
            label = self.class_to_idx[self.train_y[idx]]
        else:
            img_path = self.val_paths[idx]
            image = imread(img_path, mode="RGB")
            label = self.class_to_idx[self.val_y[idx]]

        if len(image.shape) == 2:
            print("Found grayscale image -> stacked to RGB")
            img = np.stack([image] * 3, 2)

        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    """Plot first N images and labels (only 1. image of the batch)"""
    N = 4
    awe_dataset = AWEDataset("AWE_dataset", train=True, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
        ]))
    awe_loader = DataLoader(awe_dataset, batch_size=1, shuffle=True, num_workers=1)

    fig = plt.figure()

    for i_batch, batch in enumerate(awe_loader):
        img, label = batch
        img = np.transpose(img.numpy(), (0, 2, 3, 1))

        ax = plt.subplot(1, N, i_batch + 1)
        plt.tight_layout()
        ax.set_title("Sample #{}".format(label[0]))
        ax.axis("off")
        print(img[0].shape, label[0])
        plt.imshow(img[0])

        if i_batch == N-1:
            plt.show()
            break
