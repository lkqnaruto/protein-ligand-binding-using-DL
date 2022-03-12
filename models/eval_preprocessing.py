import numpy as np
import torch
import dill
import pickle
from scipy.spatial import distance_matrix
import copy
from copy import deepcopy
import bisect
import os
import random
import sys


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========================================================================================


def save(a, file_name):
    with open(file_name, 'wb') as handle:
        dill.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ========================================================================================


def load(file_name):
    with open(file_name, 'rb') as handle:
        return dill.load(handle)


# ========================================================================================


def distance_3D(x, y, axis=None):
    diff = np.array(x) - np.array(y)
    diff = diff ** 2
    return np.sqrt(np.sum(diff, axis=axis))

# ========================================================================================


def distance_to_atoms(candidate, verts, axis=None):

    return distance_3D(candidate, verts, axis=axis)

# ========================================================================================


def one_hot(vector):

    values = vector.astype(int).tolist()
    n_values = 23
    one_hot_matrix = np.eye(n_values)[:, 1:][values]

    return one_hot_matrix


# ========================================================================================


def one_hot_3D_matrix(one_hot_matrix, atom_type_vector):

    atom_type_matrix = np.zeros((22, atom_type_vector.shape[0], atom_type_vector.shape[0]))

    assert atom_type_matrix.shape == (22, 51, 51)

    for index1 in range(atom_type_vector.shape[0]):
        index2 = 0
        while index2 < atom_type_vector.shape[0]:
            if atom_type_vector[index1].astype(int) == atom_type_vector[index2].astype(int):
                one_hot = one_hot_matrix[index1]
                atom_type_matrix[:, index1, index2] = one_hot
            else:
                two_hot = one_hot_matrix[index1] + one_hot_matrix[index2]
                atom_type_matrix[:, index1, index2] = two_hot
            index2 += 1

    return atom_type_matrix

# ========================================================================================


def dist_to_one_hot(dm):

    range_list = [0, 4, 10, 80]
    binsize = [0.2, 1, 40]
    dist_one_hot_matrix = np.zeros((29, 51, 51))

    for row_index in range(51):
        for col_index in range(51):

            idx = bisect.bisect(range_list, dm[row_index, col_index]) - 1
            bin_width = binsize[idx]
            nbins = np.ceil((dm[row_index, col_index] - range_list[idx]) / bin_width)

            total_nbins = 0
            for i in range(1, idx + 1):
                total_nbins = total_nbins + (range_list[i] - range_list[i - 1]) / binsize[i - 1]

            total_nbins += nbins
            dist_one_hot_matrix[total_nbins.astype(int), row_index, col_index] = 1

    return dist_one_hot_matrix

# ========================================================================================


def eval_30channel_preprocessing(current_pred_loc, atoms_data, seed):
    # seed_everything(seed)

    current_loc = [22] + current_pred_loc.tolist()
    indicator_matrix = np.ones((1, 51, 51))
    small_indicator_matrix = np.zeros((1, 50, 50))
    indicator_matrix[:, 1:, 1:] = small_indicator_matrix

    random_atoms_data = np.random.permutation(atoms_data)
    random_true_ligand_atoms = np.vstack([current_loc, random_atoms_data])
    temp_input_dm = distance_matrix(random_true_ligand_atoms.T[1:].T, random_true_ligand_atoms.T[1:].T)

    input_dis_onehot_matrix = dist_to_one_hot(temp_input_dm)

    input_dm = np.concatenate((input_dis_onehot_matrix, indicator_matrix), axis=0).reshape(1, 30, 51, 51)
    assert input_dm.shape == (1, 30, 51, 51)

    return input_dm


# ========================================================================================


def eval_52channel_preprocessing(current_pred_loc, atoms_data, seed):

    # seed_everything(seed)

    current_loc = [22] + current_pred_loc
    indicator_matrix = np.ones((1, 51, 51))
    small_indicator_matrix = np.zeros((1, 50 , 50))
    indicator_matrix[:, 1:, 1:] = small_indicator_matrix


    random_atoms_data = np.random.permutation(atoms_data)
    random_true_ligand_atoms = np.vstack([current_loc, random_atoms_data])
    temp_input_dm = distance_matrix(random_true_ligand_atoms.T[1:].T, random_true_ligand_atoms.T[1:].T)


    input_dis_onehot_matrix = dist_to_one_hot(temp_input_dm)

    input_dm = np.concatenate((input_dis_onehot_matrix, indicator_matrix), axis=0)
    assert input_dm.shape == (30, 51, 51)

    atom_type_vector = np.vstack([np.array([current_loc[0]]).reshape(-1, 1),
                                  (np.array(random_atoms_data).T[0]).T.reshape(np.array(random_atoms_data).shape[0], -1)]).T[0]
    one_hot_matrix = one_hot(atom_type_vector)
    input_ligand_atom_type_matrix = one_hot_3D_matrix(one_hot_matrix, atom_type_vector)
    input_dm_atoms_type = np.concatenate((input_dm, input_ligand_atom_type_matrix), axis=0).reshape(1, 52, 51, 51)
    assert input_dm_atoms_type.shape == (1, 52, 51, 51)

    return input_dm_atoms_type


# ========================================================================================


def eval_24channel_preprocessing(current_pred_loc, atoms_data, seed):


    # seed_everything(seed)

    current_loc = [22] + current_pred_loc.tolist()

    indicator_matrix = np.ones((1, 51, 51))
    small_indicator_matrix = np.zeros((1, 50 , 50))
    indicator_matrix[:, 1:, 1:] = small_indicator_matrix

    random_atoms_data = np.random.permutation(atoms_data)
    random_true_ligand_atoms = np.vstack([current_loc, random_atoms_data])
    temp_input_dm = distance_matrix(random_true_ligand_atoms.T[1:].T, random_true_ligand_atoms.T[1:].T)

    input_dm = np.concatenate((temp_input_dm.reshape(1, 51, 51), indicator_matrix), axis = 0)
    atom_type_vector = np.vstack([np.array([current_loc[0]]).reshape(-1, 1),
                                  (np.array(random_atoms_data).T[0]).T.reshape(np.array(random_atoms_data).shape[0], -1)]).T[0]
    one_hot_matrix = one_hot(atom_type_vector)
    input_ligand_atom_type_matrix = one_hot_3D_matrix(one_hot_matrix, atom_type_vector)
    input_dm_atoms_type = np.concatenate((input_dm, input_ligand_atom_type_matrix), axis=0).reshape(1, 24, 51, 51)
    assert input_dm_atoms_type.shape == (1, 24, 51, 51)

    return input_dm_atoms_type


# ========================================================================================


def eval_2channel_preprocessing(current_pred_loc, atoms_data, seed):

    # seed_everything(seed)

    current_loc = [22] + current_pred_loc.tolist()

    indicator_matrix = np.ones((1, 51, 51))
    small_indicator_matrix = np.zeros((1, 50 , 50))
    indicator_matrix[:, 1:, 1:] = small_indicator_matrix

    random_atoms_data = np.random.permutation(atoms_data)
    random_input_ligand_atoms = np.vstack([current_loc, random_atoms_data])

    temp_input_dm = distance_matrix(random_input_ligand_atoms.T[1:].T, random_input_ligand_atoms.T[1:].T)
    input_dm = np.concatenate((temp_input_dm.reshape(1, 51, 51), indicator_matrix), axis=0).reshape(1, 2, 51, 51)

    assert input_dm.shape == (1, 2, 51, 51)

    return input_dm


# ========================================================================================


if __name__ == '__main__':

    # input_data_30channel = data_preprocessing_30channel('/Users/keqiaoli/Desktop/protein_ligand_test/0615_input_data_50000','train')
    # print(input_data_30channel.shape)
    # save(input_data_52channel, '/0615_input_data_50000/train_input_data_52channel.pickle')

    for index in range(3):

        seed_everything(1989)
        pred_loc = torch.tensor([1, 2, 3])
        atoms_data = np.random.normal(20, 5, (50,3))
        atom_type = np.random.randint(1,22, (50, 1))
        atoms_data = torch.tensor(np.vstack([atom_type.reshape(1, 50), atoms_data.T]).T)
        print(atoms_data[:3, :])
        print(atoms_data.shape)
        seed = 1989
        dm = eval_2channel_preprocessing(pred_loc, atoms_data, seed)
        # print(dm.shape)
        print(dm[0, 2, :])

