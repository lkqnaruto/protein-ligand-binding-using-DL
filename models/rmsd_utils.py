import scipy
import torch
import numpy as np
import sys
from . import Preprocessing
# data_preprocessing_2channel, data_preprocessing_24channel_multi, \
#     data_preprocessing_30channel, data_preprocessing_52channel
import math
import time
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

# ================================================================================================


def rmsd(predict, true):

    assert np.array(predict).shape == np.array(true).shape == (5,4)

    dis_S = distance_3D(np.array(predict).T[1:].T[0], np.array(true).T[1:].T[0])
    dm = distance_matrix(np.array(predict).T[1:].T[1:], np.array(true).T[1:].T[1:])
    row_ind1, col_ind1 = linear_sum_assignment(dm)
    dm2 = np.power(dm, 2)
    dis_final_min = dm2[row_ind1, col_ind1].sum()
    dis_total_min = np.sqrt((dis_final_min + dis_S**2) / 5)

    return dis_total_min


# ======================================================================================================================


def extract_dis_vectors(dm):

    assert dm.shape == (55, 55)
    dis_vector_Os = [dm[index, 5:].tolist() for index in range(1,5)]

    return dis_vector_Os


# ================================================================================================


def distance_to_loc(vector_d, verts):
    vector_d = np.asarray(vector_d)
    verts = np.asarray(verts)
    A = 2 * (verts[1:] - verts[0])
    Y = vector_d[0] ** 2 - vector_d[1:] ** 2 + np.sum(verts[1:] ** 2 - verts[0] ** 2, axis=1)
    loc = np.matmul(np.matmul(np.linalg.pinv(np.matmul(A.T, A)), A.T), Y)

    return loc

# ======================================================================================================================


def distance_3D(x, y, axis=None):
    diff = np.array(x) - np.array(y)
    diff = diff ** 2
    return np.sqrt(np.sum(diff, axis=axis))


# ======================================================================================================================

# def one_step_prediction(pred_dist_vector, verts):
#
#     pred_loc = distance_to_loc(pred_dist_vector[1:].cpu().detach().numpy(), verts)
#
#     return pred_loc
#
#
# # ======================================================================================================================
#
# def preprocessing_two_step(pred_loc, current_true_loc, selected_ligand_atom_pair):
#
#     atoms_data = selected_ligand_atom_pair[current_true_loc]
#     input_dm = eval_24channel_preprocessing(pred_loc, atoms_data)
#
#     return input_dm
#

# ======================================================================================================================

def get_rmsd_Os(y_fake, random_env_atom_data, ligand_key, true_ligands):

    dis_vector_ligand_Os = extract_dis_vectors(y_fake)

    temp_ligand_loc_Os = [distance_to_loc(dis_vector_ligand_Os[j], random_env_atom_data.T[1:].T).tolist() for j in
                          range(4)]

    pred_ligand_loc_S = distance_to_loc(y_fake.cpu()[0, 5:], random_env_atom_data.T[1:].T)
    dists_S = distance_3D(pred_ligand_loc_S, np.array(ligand_key)[1:])

    estimated_ligands_locs = np.vstack([pred_ligand_loc_S, temp_ligand_loc_Os])

    estimated_ligands_data = np.vstack([[22, 20, 20, 20, 20], estimated_ligands_locs.T]).T

    dis_final_min = rmsd(estimated_ligands_data, true_ligands)

    return dis_final_min, dists_S

# ======================================================================================================================


def get_prediction(key, env_points, ligand_atom_pair, G_model):

    all_points_data = env_points[key]
    point_step_index = np.random.randint(1, 100)
    # print(point_step_index)
    starting_points = all_points_data[0][point_step_index]  # 0 is the path index since only one path generated
    # print("staring points are:", starting_points)
    data_24channel, starting_ligands_data, random_atom_data = Preprocessing.data_preprocessing_24channel_multi(starting_points,
                                                                                                 ligand_atom_pair,
                                                                                                 key)

    true_atoms_data = ligand_atom_pair[key]['ligand_atoms']
    true_ligands_data = np.vstack([key, true_atoms_data])

    starting_dist_min = rmsd(starting_ligands_data, true_ligands_data)


    x = torch.tensor(data_24channel[:, :, :55]).unsqueeze(0).float()
    y = torch.tensor(data_24channel[:, :, 55:]).unsqueeze(0).float()
    assert x.shape == (1, 24, 55, 55)
    assert y.shape == (1, 24, 55, 55)

    x[0, 0, :, :] = (x[0, 0, :, :] / 80) * 2 - 1
    x = x.to(config.DEVICE)
    with torch.no_grad():
        y_fake = G_model(x)                           # y_fake: 4 dimensions
    pred_y = ((y_fake[0, 0, :, :] + 1) / 2) * 80  # remove normalization and extract the distance matrix
    assert pred_y.shape == (55, 55)


    dis_final_min, dists_S = get_rmsd_Os(pred_y, random_atom_data, key, true_ligands_data)
    # dists_onestep.append(dis_final_min.tolist())

    # return dists_onestep, pred_loc_onestep_list, dists_twostep, pred_loc_twostep_list, starting_dists, dists_S
    return dis_final_min, starting_dist_min, dists_S


# ======================================================================================================================

def get_input_dis_matrix(target_key, all_points, selected_ligand_atom_pair):
    print([*all_points.keys()][0] == target_key)
    all_points_data = all_points[target_key]
    point_step_index = np.random.randint(1, 100)
    # print(point_step_index)
    starting_points = all_points_data[0][point_step_index]  # 0 is the path index since only one path generated
    # print("staring points are:", starting_points)
    data_24channel, starting_ligands_data, random_atom_data = Preprocessing.data_preprocessing_24channel_multi(starting_points,
                                                                                                 selected_ligand_atom_pair,
                                                                                                 target_key)

    true_atoms_data = selected_ligand_atom_pair[target_key]['ligand_atoms']
    true_ligands_data = np.vstack([target_key, true_atoms_data])

    starting_dist_min = rmsd(starting_ligands_data, true_ligands_data)

    x = torch.tensor(data_24channel[:, :, :55]).unsqueeze(0).float()
    y = torch.tensor(data_24channel[:, :, 55:]).unsqueeze(0).float()
    assert x.shape == (1, 24, 55, 55)
    assert y.shape == (1, 24, 55, 55)

    return x, y, starting_dist_min, random_atom_data, true_ligands_data
