import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import pickle as pickle
import torch
from data.Preprocessing import data_preprocessing_2channel, seed_everything, data_preprocessing_24channel_multi, data_preprocessing_52channel, data_preprocessing_30channel



def save(a, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(a, handle, protocol=4)

# ========================================================================================

def load(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)

# ========================================================================================


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        #
        # self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        # self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        # self.A_size = len(self.A_paths)  # get the size of dataset A
        # self.B_size = len(self.B_paths)  # get the size of dataset B
        # btoA = self.opt.direction == 'BtoA'
        # input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        # output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        # self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

        self.dir =  opt.dataroot
        print("loading: ", self.dir)
        self.all_points = load(self.dir + '/' + 'train_all_points_dict.pickle')

        self.selected_ligand_atom_pair = load(self.dir + '/' + 'train_selected_ligand_atom_pair.pickle')

        self.ligand_loc_key = torch.tensor([*self.all_points.keys()])
        self.len = len(self.ligand_loc_key)


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        # if self.opt.serial_batches:   # make sure index is within then range
        #     index_B = index % self.B_size
        # else:   # randomize the index for domain B to avoid fixed pairs.
        #     index_B = random.randint(0, self.B_size - 1)
        # B_path = self.B_paths[index_B]
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        # # apply image transformation
        # A = self.transform_A(A_img)
        # B = self.transform_B(B_img)
        A_path = self.dir
        B_path = self.dir
        target_keys = self.ligand_loc_key[index]
        # print(target_keys)
        #         print(index)
        # data_2channel = data_preprocessing_2channel(self.all_points, self.selected_ligand_atom_pair, target_keys)
        # data_52channel = data_preprocessing_52channel(self.all_points, self.selected_ligand_atom_pair, target_keys)
        data_24channel = data_preprocessing_24channel_multi(self.all_points, self.selected_ligand_atom_pair,
                                                            target_keys)
        # data_30channel = data_preprocessing_30channel(self.all_points, self.selected_ligand_atom_pair, target_keys)

        A = data_24channel[:, :, :55]
        B = data_24channel[:, :, 55:]
        A[0, :, :] = (A[0, :, :] / 80) * 2 - 1
        B[0, :, :] = (B[0, :, :] / 80) * 2 - 1


        # print("-------------------loading---------------------")
        # print(A.shape, B.shape)

        return {'A': torch.tensor(A).float(), 'B': torch.tensor(B).float(), 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.len #(self.A_size, self.B_size)
