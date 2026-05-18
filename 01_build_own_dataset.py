import os
from torch.utils.data import Dataset, ConcatDataset # attention .data
from PIL import Image

# Part1 build own dataset
class MyData(Dataset): # must inherit Dataset
    root_dir = r'data\hymenoptera_data\train'
    label_dir = 'ants'
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.imgs_list = os.listdir(self.path)
    def __getitem__(self, idx):
        img_name = self.imgs_list[idx]
        img_item_path = os.path.join(self.root_dir,
                                     self.label_dir,
                                     img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    def __len__(self):
        return len(self.imgs_list)

# Part2 apply
root_dir = r'data\hymenoptera_data\train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

sample_img, sample_label = ants_dataset[0]
# sample_img.show()

merge_dataset = ConcatDataset([ants_dataset, bees_dataset]) # attention how dataset merge
img1, label1 = merge_dataset[123]
img2, label2 = merge_dataset[124]

img1.show()
img2.show()