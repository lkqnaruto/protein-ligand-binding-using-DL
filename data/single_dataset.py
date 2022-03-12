from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import pickle

# ========================================================================================


def save(a, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(a, handle, protocol=4)

# ========================================================================================

def load(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)

# ========================================================================================

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        # input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # self.transform = get_transform(opt, grayscale=(input_nc == 1))
        #

        self.dir =  opt.dataroot
        print("loading: ", self.dir)
        self.all_points = load(self.dir + '/' + 'test_all_points_dict.pickle')
        self.selected_ligand_atom_pair = load(self.dir + '/' + 'test_selected_ligand_atom_pair.pickle')

        self.ligand_loc_key = torch.tensor([*self.all_points.keys()])
        # self.ligand_loc_key = [*self.selected_ligand_atom_pair.keys()]
        self.len = len(self.ligand_loc_key)



    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        # A_path = self.A_paths[index]
        # A_img = Image.open(A_path).convert('RGB')
        # A = self.transform(A_img)

        A = self.ligand_loc_key[index]


        key_path = self.dir

        return {'key': A, 'A_paths': key_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.len
