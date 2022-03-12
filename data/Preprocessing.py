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

    assert atom_type_matrix.shape == (22, atom_type_vector.shape[0], atom_type_vector.shape[0])

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


def data_preprocessing_52channel(all_points, selected_ligand_atom_pair, key):


    indicator_matrix = np.ones((1, 51, 51))
    small_indicator_matrix = np.zeros((1, 50 , 50))
    indicator_matrix[:, 1:, 1:] = small_indicator_matrix

    key_ligand = tuple(key.numpy())
    starting_ligand = key_ligand
    atoms_data = selected_ligand_atom_pair[key_ligand]
    random_atoms_data = np.random.permutation(atoms_data)
    random_true_ligand_atoms = np.vstack([starting_ligand, random_atoms_data])
    temp_true_dm = distance_matrix(random_true_ligand_atoms.T[1:].T, random_true_ligand_atoms.T[1:].T)
    atoms_dm = temp_true_dm[1:, 1:]

    true_dis_onehot_matrix = dist_to_one_hot(temp_true_dm)
    print(true_dis_onehot_matrix.shape)
    true_dm = np.concatenate((true_dis_onehot_matrix, indicator_matrix), axis=0)
    assert true_dm.shape == (30, 51, 51)

    atom_type_vector = np.vstack([np.array([key_ligand[0]]).reshape(-1, 1),
                                  (np.array(random_atoms_data).T[0]).T.reshape(np.array(random_atoms_data).shape[0], -1)]).T[0]
    one_hot_matrix = one_hot(atom_type_vector)
    true_ligand_atom_type_matrix = one_hot_3D_matrix(one_hot_matrix, atom_type_vector)
    true_dm_atoms_type = np.concatenate((true_dm, true_ligand_atom_type_matrix), axis=0)
    assert true_dm_atoms_type.shape == (52, 51, 51)

    for key_path in range(1):

        generated_loc = all_points[key_ligand][key_path][-1][1:]
        fake_dm_vector = distance_to_atoms(generated_loc, random_atoms_data.T[1:].T, axis=1)
        temp_fake_dm = np.zeros((51, 51))
        temp_fake_dm[1:, 1:] = deepcopy(atoms_dm)
        temp_fake_dm[0, 1:] = fake_dm_vector.copy()
        temp_fake_dm[1:, 0] = fake_dm_vector.T.copy()

        fake_dis_onehot_matrix = dist_to_one_hot(temp_fake_dm)
        fake_dm = np.concatenate((fake_dis_onehot_matrix, indicator_matrix), axis=0)
        fake_dm_atoms_type = np.concatenate((fake_dm, true_ligand_atom_type_matrix), axis=0)
        assert fake_dm_atoms_type.shape == (52, 51, 51)
        temp = np.concatenate((fake_dm_atoms_type, true_dm_atoms_type ), axis=2)
        assert temp.shape == (52, 51, 102)
        # print(input_data)

    return temp

# ========================================================================================

def data_preprocessing_30channel(all_points, selected_ligand_atom_pair, key):


    indicator_matrix = np.ones((1, 51, 51))
    small_indicator_matrix = np.zeros((1, 50 , 50))
    indicator_matrix[:, 1:, 1:] = small_indicator_matrix

    key_ligand = tuple(key.numpy())
    starting_ligand = key_ligand
    atoms_data = selected_ligand_atom_pair[key_ligand]
    random_atoms_data = np.random.permutation(atoms_data)
    random_true_ligand_atoms = np.vstack([starting_ligand, random_atoms_data])
    temp_true_dm = distance_matrix(random_true_ligand_atoms.T[1:].T, random_true_ligand_atoms.T[1:].T)
    atoms_dm = temp_true_dm[1:, 1:]

    true_dis_onehot_matrix = dist_to_one_hot(temp_true_dm)
    # print(true_dis_onehot_matrix.shape)
    true_dm = np.concatenate((true_dis_onehot_matrix, indicator_matrix), axis=0)
    assert true_dm.shape == (30, 51, 51)

    # atom_type_vector = np.vstack([np.array([key_ligand[0]]).reshape(-1, 1),
    #                               (np.array(random_atoms_data).T[0]).T.reshape(np.array(random_atoms_data).shape[0], -1)]).T[0]
    # one_hot_matrix = one_hot(atom_type_vector)
    # true_ligand_atom_type_matrix = one_hot_3D_matrix(one_hot_matrix, atom_type_vector)
    # true_dm_atoms_type = np.concatenate((true_dm, true_ligand_atom_type_matrix), axis=0)
    # assert true_dm_atoms_type.shape == (52, 51, 51)

    for key_path in range(1):

        generated_loc = all_points[key_ligand][key_path][-1][1:]
        fake_dm_vector = distance_to_atoms(generated_loc, random_atoms_data.T[1:].T, axis=1)
        temp_fake_dm = np.zeros((51, 51))
        temp_fake_dm[1:, 1:] = deepcopy(atoms_dm)
        temp_fake_dm[0, 1:] = fake_dm_vector.copy()
        temp_fake_dm[1:, 0] = fake_dm_vector.T.copy()

        fake_dis_onehot_matrix = dist_to_one_hot(temp_fake_dm)
        fake_dm = np.concatenate((fake_dis_onehot_matrix, indicator_matrix), axis=0)
        # fake_dm_atoms_type = np.concatenate((fake_dm, true_ligand_atom_type_matrix), axis=0)
        # assert fake_dm_atoms_type.shape == (52, 51, 51)
        temp = np.concatenate((fake_dm, true_dm), axis=2)
        assert temp.shape == (30, 51, 102)
        # print(input_data)

    return temp



# ========================================================================================


def data_preprocessing_24channel_multi(all_points, selected_ligand_atom_pair, key):


    indicator_matrix = np.ones((1, 55, 55))
    small_indicator_atoms_matrix = np.zeros((1, 50 , 50))
    small_indicator_ligand_matrix = np.zeros((1, 5 , 5))
    indicator_matrix[:, 5:, 5:] = small_indicator_atoms_matrix
    indicator_matrix[:, :5, :5] = small_indicator_ligand_matrix


    key_ligand = tuple(key.numpy())
    starting_ligand = key_ligand
    atoms_data = selected_ligand_atom_pair[key_ligand]['env_atoms']
    ligand_atoms_data = selected_ligand_atom_pair[key_ligand]['ligand_atoms']


    random_atoms_data = np.random.permutation(atoms_data)
    random_true_ligand_atoms = np.vstack([starting_ligand, ligand_atoms_data, random_atoms_data])
    temp_true_dm = distance_matrix(random_true_ligand_atoms.T[1:].T, random_true_ligand_atoms.T[1:].T)
    atoms_dm = temp_true_dm[1:, 1:]
    true_dm = np.concatenate((temp_true_dm.reshape(1, 55, 55), indicator_matrix), axis = 0)
    atom_type_vector = np.vstack([np.array([key_ligand[0]]).reshape(-1, 1),
                                  np.array([20, 20, 20, 20]).reshape(-1, 1),
                                  (np.array(random_atoms_data).T[0]).T.reshape(np.array(random_atoms_data).shape[0], -1)]).T[0]
    one_hot_matrix = one_hot(atom_type_vector)
    true_ligand_atom_type_matrix = one_hot_3D_matrix(one_hot_matrix, atom_type_vector)
    true_dm_atoms_type = np.concatenate((true_dm, true_ligand_atom_type_matrix), axis=0)
    assert true_dm_atoms_type.shape == (24, 55, 55)

    for key_path in range(1):
        
        point_step_index = np.random.randint(1,100)
        generated_locs = all_points[key_ligand][key_path][point_step_index]
        assert np.array(generated_locs).shape == (5, 4)
        random_fake_ligand_atoms = np.vstack([generated_locs, random_atoms_data])
        assert random_fake_ligand_atoms.shape == (55, 4)
        temp_fake_dm = distance_matrix(random_fake_ligand_atoms.T[1:].T, random_fake_ligand_atoms.T[1:].T)

        fake_dm = np.concatenate((temp_fake_dm.reshape(1, 55, 55), indicator_matrix), axis = 0)
        fake_dm_atoms_type = np.concatenate((fake_dm, true_ligand_atom_type_matrix), axis=0)
        assert fake_dm_atoms_type.shape == (24, 55, 55)
        temp = np.concatenate((fake_dm_atoms_type, true_dm_atoms_type), axis=2)
        assert temp.shape == (24, 55, 110)
    return temp


# ========================================================================================


def data_preprocessing_2channel(all_points, selected_ligand_atom_pair, key):

    temp = None
    indicator_matrix = np.ones((1, 51, 51))
    small_indicator_matrix = np.zeros((1, 50 , 50))
    indicator_matrix[:, 1:, 1:] = small_indicator_matrix

    key_ligand = tuple(key.numpy())
    starting_ligand = key_ligand
    atoms_data = selected_ligand_atom_pair[key_ligand]
    random_atoms_data = np.random.permutation(atoms_data)
    random_true_ligand_atoms = np.vstack([starting_ligand, random_atoms_data])

    temp_true_dm = distance_matrix(random_true_ligand_atoms.T[1:].T, random_true_ligand_atoms.T[1:].T)
    atoms_dm = temp_true_dm[1:, 1:]
    true_dm = np.concatenate((temp_true_dm.reshape(1, 51, 51), indicator_matrix), axis=0)

    assert true_dm.shape == (2, 51, 51)

    for key_path in range(1):

        point_step_index = np.random.randint(1,100)
        generated_loc = all_points[key_ligand][key_path][point_step_index][1:]
        fake_dm_vector = distance_to_atoms(generated_loc, random_atoms_data.T[1:].T, axis=1)
        temp_fake_dm = np.zeros((51, 51))
        temp_fake_dm[1:, 1:] = atoms_dm
        temp_fake_dm[0, 1:] = fake_dm_vector
        temp_fake_dm[1:, 0] = fake_dm_vector.T

        # TODO check deepcopy result when have time
        # temp_fake_dm[1:, 1:] = deepcopy(atoms_dm)
        # temp_fake_dm[0, 1:] = fake_dm_vector.copy()
        # temp_fake_dm[1:, 0] = fake_dm_vector.T.copy()

        fake_dm = np.concatenate((temp_fake_dm.reshape(1, 51, 51), indicator_matrix), axis = 0)
        # fake_dm = temp_fake_dm
        # assert true_dm.shape == (1, 2, 51, 51)
        # assert fake_dm.shape == (1, 2, 51, 51)
        # temp = np.concatenate((true_dm.reshape(1, 1, 51, 51), fake_dm.reshape(1, 1, 51, 51)), axis=3)
        temp = np.concatenate((fake_dm, true_dm), axis=2)
        assert temp.shape == (2, 51, 102)
        # input_data = np.concatenate((input_data, temp), axis=0)
        # count+=1
    # input_data = np.delete(input_data, 0, 0)

    return temp


def data_preprocessing_24channel_multi_distill(starting_ligand, atoms, true_ligand):



    numAtoms, _= np.array(atoms).shape
    numLigandAtoms, _ = np.array(true_ligand).shape
    # print(true_ligand.shape, starting_ligand, atoms.shape)
    dims = numAtoms + numLigandAtoms
    assert np.array(true_ligand).shape == starting_ligand.shape


    indicator_matrix = np.ones((1, dims, dims))
    small_indicator_atoms_matrix = np.zeros((1, numAtoms, numAtoms))
    small_indicator_ligand_matrix = np.zeros((1, numLigandAtoms, numLigandAtoms))
    indicator_matrix[:, numLigandAtoms:, numLigandAtoms:] = small_indicator_atoms_matrix
    indicator_matrix[:, :numLigandAtoms, :numLigandAtoms] = small_indicator_ligand_matrix

    # key_ligand = tuple(key.numpy())
    # starting_ligand = key_ligand
    atoms_data = atoms                  # atoms:   List
    ligand_atoms_data = true_ligand     # true_ligand:     List

    random_atoms_data = np.random.permutation(atoms_data)
    random_true_ligand_atoms = np.vstack([true_ligand, random_atoms_data])
    temp_true_dm = distance_matrix(random_true_ligand_atoms.T[1:].T, random_true_ligand_atoms.T[1:].T)
    # atoms_dm = temp_true_dm[1:, 1:]
    assert temp_true_dm.shape == (dims, dims)

    true_dm = np.concatenate((temp_true_dm.reshape(1, dims, dims), indicator_matrix), axis=0)
    assert true_dm.shape == (2, dims, dims)

    atom_type_vector = np.vstack([[[22]]*numLigandAtoms,
                                  (np.array(random_atoms_data).T[0]).T.reshape(np.array(random_atoms_data).shape[0],
                                                                               -1)]).T[0]


    one_hot_matrix = one_hot(atom_type_vector)
    true_ligand_atom_type_matrix = one_hot_3D_matrix(one_hot_matrix, atom_type_vector)
    true_dm_atoms_type = np.concatenate((true_dm, true_ligand_atom_type_matrix), axis=0)
    assert true_dm_atoms_type.shape == (24, dims, dims)



    # for key_path in range(1):
    # point_step_index = np.random.randint(1, 100)
    # generated_locs = all_points[key_ligand][key_path][point_step_index]
    # assert np.array(generated_locs).shape == (5, 4)
    random_fake_ligand_atoms = np.vstack([starting_ligand, random_atoms_data])
    assert random_fake_ligand_atoms.shape == (dims, 4)
    temp_fake_dm = distance_matrix(random_fake_ligand_atoms.T[1:].T, random_fake_ligand_atoms.T[1:].T)

    fake_dm = np.concatenate((temp_fake_dm.reshape(1, dims, dims), indicator_matrix), axis=0)
    fake_dm_atoms_type = np.concatenate((fake_dm, true_ligand_atom_type_matrix), axis=0)
    assert fake_dm_atoms_type.shape == (24, dims, dims)
    # temp = np.concatenate((fake_dm_atoms_type, true_dm_atoms_type), axis=2)
    # assert temp.shape == (24, dims, 2*dims)



    return fake_dm_atoms_type, true_dm_atoms_type


# ========================================================================================


if __name__ == '__main__':

    input_data_52channel = data_preprocessing_2channel('/0615_input_data_50000','train')
    print(input_data_52channel.shape)
    save(input_data_52channel, '/0615_input_data_50000/train_input_data_52channel.pickle')



