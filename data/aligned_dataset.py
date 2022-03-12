import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import pickle as pickle
import torch
from data.Preprocessing import data_preprocessing_2channel, seed_everything, data_preprocessing_24channel_multi, data_preprocessing_52channel, data_preprocessing_30channel, \
    data_preprocessing_24channel_multi_distill
import numpy as np
import torch.nn as nn

def save(a, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(a, handle, protocol=4)

# ========================================================================================

def load(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)

# ========================================================================================

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    # def __init__(self, opt):
    def __init__(self, dir):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # BaseDataset.__init__(self, opt)
        BaseDataset.__init__(self, dir)
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        # assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        # self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        # self.dir =  opt.dataroot
        self.dir =  dir
        # print("loading: ", self.dir)
        self.max_dims = 200
        # self.all_points = load(self.dir + '/' + 'ligand_env_coords.pickle')
        self.ligand_atoms_pair = load(self.dir  + 'ligand_env_coords.pickle')

        # self.selected_ligand_atom_pair = load(self.dir + '/' + 'train_selected_ligand_atom_pair.pickle')
        #
        # self.ligand_loc_key = torch.tensor([*self.all_points.keys()])
        # self.len = len(self.ligand_loc_key)
        self.len = len(self.ligand_atoms_pair)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        # AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))
        #
        # # apply the same transform to both A and B
        # transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        #
        # A = A_transform(A)
        # B = B_transform(B)
        pdbid, ligand_name, ligand, atoms= self.ligand_atoms_pair[index]
        ligand_loc_list = [ele[1:] for ele in ligand]
        center_loc = np.mean(np.array(ligand_loc_list), axis=0)
        assert center_loc.shape == (3,)
        start_center_loc = self.path_generator(atoms, center_loc.tolist())
        starting_ligand_loc = np.array(ligand_loc_list) + np.array(start_center_loc) - center_loc
        starting_ligand = np.vstack([[22]*len(ligand), starting_ligand_loc.T]).T

        # data_24channel = data_preprocessing_24channel_multi(starting_ligand_loc, atoms,
        #                                                     ligand)
        tempFakeA, tempTrueB = data_preprocessing_24channel_multi_distill(starting_ligand, atoms, ligand)

        assert tempFakeA.shape[2] <=200
        assert tempTrueB.shape[2] <= 200

        if tempFakeA.shape[2] % 2 == 0:
            padding = self.max_dims - tempFakeA.shape[1]
            m = nn.ZeroPad2d(padding // 2)
        else:
            assert tempFakeA.shape[2] % 2 == 1
            padding = self.max_dims - tempFakeA.shape[1]
            m = nn.ZeroPad2d((padding//2, padding//2+1, padding//2, padding//2+1))

        A, B = m(torch.tensor(tempFakeA)), m(torch.tensor(tempTrueB))
        A[0, :, :] = (A[0, :, :] / (torch.max(A)+1)) * 2 - 1
        B[0, :, :] = (B[0, :, :] / (torch.max(A)+1)) * 2 - 1

        assert A.shape == (24, 250, 250)
        assert B.shape == (24, 250, 250)
        """
        
        A_path = self.dir
        B_path = self.dir
        target_keys = self.ligand_loc_key[index]
        # print(target_keys)
        #         print(index)
        # data_2channel = data_preprocessing_2channel(self.all_points, self.selected_ligand_atom_pair, target_keys)
        # data_52channel = data_preprocessing_52channel(self.all_points, self.selected_ligand_atom_pair, target_keys)
        
        # data_30channel = data_preprocessing_30channel(self.all_points, self.selected_ligand_atom_pair, target_keys)

        A = data_24channel[:, :, :55]
        B = data_24channel[:, :, 55:]
        A[0, :, :] = (A[0, :, :] / 80) * 2 - 1
        B[0, :, :] = (B[0, :, :] / 80) * 2 - 1

        # print("-------------------loading---------------------")
        # print(A.shape, B.shape)
        """
        # return {'A': torch.tensor(A).float(), 'B': torch.tensor(B).float(), 'A_paths': A_path, 'B_paths': B_path}
        return {'A': torch.tensor(A).float(), 'B': torch.tensor(B).float(), 'A_paths': "None", 'B_paths': "None",
                'Ligand_length': len(ligand) , 'env_atoms_length': len(atoms) }
        # return pdbid, ligand_name, ligand, atoms


    def __len__(self):
        """Return the total number of images in the dataset."""
        # return len(self.AB_paths)
        return self.len

    # ==============================================================================

    def generate_the_next_candidate_point(self, current_timestep_point: list, dir: str) -> tuple:
        next_candidate = np.array(current_timestep_point)
        increment = 0.5
        mu, sigma = 0, 0.1

        if dir == 'right':
            next_candidate[0] = next_candidate[0] + increment + np.random.normal(mu, sigma, 1)
        if dir == 'left':
            next_candidate[0] = next_candidate[0] - increment + np.random.normal(mu, sigma, 1)
        if dir == 'forward':
            next_candidate[1] = next_candidate[1] - increment + np.random.normal(mu, sigma, 1)
        if dir == 'backward':
            next_candidate[1] = next_candidate[1] + increment + np.random.normal(mu, sigma, 1)
        if dir == 'up':
            next_candidate[2] = next_candidate[2] + increment + np.random.normal(mu, sigma, 1)
        if dir == 'down':
            next_candidate[2] = next_candidate[2] - increment + np.random.normal(mu, sigma, 1)

        return tuple(next_candidate)

    # ==============================================================================


    def distance_3D(self, x, y, axis=None):
        diff = np.array(x) - np.array(y)
        diff = diff ** 2
        return np.sqrt(np.sum(diff, axis=axis))

    # ==============================================================================

    def distance_to_atoms(self, candidate, verts, axis=None):
        # return tuple(distance_3D(candidate, verts, axis=axis))
        return self.distance_3D(candidate, verts, axis=axis)

    # ==============================================================================

    def softmax(self, x):
        """ applies softmax to an input x"""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    # ========================================================================================

    def path_generator(self, atoms_locations, ligand_starting_loc):

        next_loc = None
        verts_loc = np.array(atoms_locations).T[1:].T
        current_loc = ligand_starting_loc
        NumberofPoints = np.random.randint(1, 100)

        for step_index in range(1, NumberofPoints + 1):

            candidate_d_dict = {}
            T = 700 / (step_index + 1)
            direction_list = ['right', 'left', 'forward', 'backward', 'up', 'down']
            denominator_ss_list = []
            for direction in direction_list:
                # step with randomness
                candidate_loc = self.generate_the_next_candidate_point(current_loc, direction)
                candidate_d_dict[direction] = candidate_loc
                distance = self.distance_to_atoms(list(candidate_loc), verts_loc, axis=1)
                ss = np.sum(np.asarray(distance) ** 2)
                denominator_ss_list.append(ss / T)

            p_list = []
            for a in denominator_ss_list:
                diff = np.array(denominator_ss_list) - a
                temp = np.sum(np.exp(np.array(diff)))
                p_list.append(1 / temp)

            # p_list = softmax(np.array(denominator_ss_list))
            direction = np.random.choice(direction_list, p=p_list)
            next_loc = candidate_d_dict[direction]

            current_loc = list(next_loc)

        return next_loc


if __name__=="__main__":
    dir = ""
    data = AlignedDataset(dir)
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=2,
        shuffle=False)
    a = next(iter(dataloader))
    print(a)