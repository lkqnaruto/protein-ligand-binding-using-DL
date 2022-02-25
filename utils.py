import os

import numpy as np
import pandas as pd
import sys
import json
from collections import deque, defaultdict


def list_files_recursive(input_path):
    '''
    List all pdb files under current location

    :param input_path: '.'
    :return: a list of file names ['./data/xxxx.pdb','./data/xxxx.pdb']
    '''

    file_list = list()
    dir_list = list()
    if os.path.isfile(input_path):
        file_list.append(input_path)
    elif os.path.isdir(input_path):
        dir_list.append(input_path)
    else:
        raise RuntimeError("Input path must be a file or directory: " + input_path)
    while len(dir_list) > 0:
        dir_name = dir_list.pop()
        # print("Processing directory " + dir_name)
        dir = os.listdir(dir_name)
        for item in dir:
            input_filename = dir_name
            if not input_filename.endswith("/"):
                input_filename += "/"
            input_filename += item
            #print("Checking item " + input_filename)
            if os.path.isfile(input_filename):
                file_list.append(input_filename)
            elif os.path.isdir(input_filename):
                dir_list.append(input_filename)
    return file_list


def read_pdb(file_name):
    '''
    Read protein env
    ! discard atom H;
    ! Skip alternative positions

    :param file_name: './data/xxxx.pdb'
    :return: [[label,x,y,z],...]
    '''
    data = []
    with open (file_name) as f:
        for line in f:
            if line[0:6]=='ATOM  ':
                atomName = line[12:16].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                altLoc = line[16]
                atomSym = line[76:78].strip()
                resName = line[17:20]
                if atomSym != 'H':
                    # The corresponding atom channel, given residual name and atom type
                    label = label_atom(resName, atomName)
                    if altLoc == ' ' or altLoc == 'A':  # skip alternate position
                        data.append([label,x,y,z])
    return data


# def read_ligand(file_name, name):
#     data = []
#     with open(file_name) as f:
#         for line in f:
#             if line[0:6] == 'HETATM':
#                 atomName = line[12:16].strip()
#                 x = float(line[30:38].strip())
#                 y = float(line[38:46].strip())
#                 z = float(line[46:54].strip())
#                 altLoc = line[16]
#                 if altLoc == ' ' or altLoc == 'A':  # skip alternate position
#                     if atomName == name:
#                         data.append([22, x, y, z])
#     return data


def read_ligand_simple(file_name, name, chain, pos):
    '''
    Read 1-molecule ligand from pdb

    :param (file_name, name, chain, pos): './data/xxxx.pdb', ligand name, chain, molecule position
    :return: ???
        1
        2
        (list(set(data)), data)
    '''
    data = []
    with open(file_name) as f:
        for line in f:
            altLoc = line[16]
            if line[0:6] == 'HETATM' and (altLoc == ' ' or altLoc == 'A'):
                ligand_name = line[17:20].strip()
                chainID, position, atomSym = line[21], line[22:26].strip(), line[76:78].strip() # atom type
                x, y, z = line[30:38].strip(), line[38:46].strip(), line[46:54].strip()
                if chainID == chain and position == pos:
                    if ligand_name != name: #???
                        return 1
                    data.append((atomSym, [float(i) for i in [x,y,z]]))

        if data == []:
            return 2
        return data

def f_read_pdb_line(idx, lines):
    '''
    :param idx:
    :param lines:
    :return:
    '''

    line = lines[idx]
    protein_ligand = line[0:6]
    ligand_name = line[17:20].strip()
    altLoc = line[16]
    chainID, position = line[21], line[22:26].strip()
    atomSym = line[76:78].strip()
    x,y,z = line[30:38].strip(), line[38:46].strip(), line[46:54].strip()
    return protein_ligand, ligand_name, chainID, position, altLoc, atomSym, (x,y,z)

def f_molecule_not_end(idx,lines, chain, pos):
    '''

    :param ligand_name:
    :param chainID:
    :param position:
    :param altLoc:
    :return:
    '''
    next_protein_ligand, next_ligand_name, next_chainID, next_position, next_altLoc, _,_ = f_read_pdb_line(idx + 1, lines)
    next_2_protein_ligand, next_2_ligand_name, next_2_chainID, next_2_position, next_2_altLoc, _,_ = f_read_pdb_line(
        idx + 2, lines)
    next = next_position == pos and (next_protein_ligand in set(['HETATM', 'ATOM  '])) and next_chainID == chain
    next2 = next_2_position == pos and next_2_chainID == chain and next_2_protein_ligand in set(['HETATM', 'ATOM  '])
    out = not(next and next2)
    return not out

def read_ligand_complex(file_name, name, chain, pos):
    '''
    Read multi-molecule ligand from pdb

    :param file_name: './abcd.pdb'
    :param name: ligand name 'NGA NAG'
    :param chain: chain 'A','B',...
    :param pos: position '1','2',...

    :return:
        1
        (list(set(data)), data): (unique atom type), (ligand atoms)
    '''

    tmp_name_list = name.split(' ')  # ['ABC', 'BCD', 'EFG', 'FGH']
    numLigands = len(tmp_name_list)
    name_list = deque(tmp_name_list)
    data = []
    index_list = []
    incomplete_indicator = False

    with open(file_name) as f:
        lines = f.readlines()
        for i in range(numLigands):
            target_ligand = name_list[0]

            assert len(lines[0]) == len(lines[1]) #???
            for index, line in enumerate(lines):
                protein_ligand, ligand_name, chainID, position, altLoc, _, axes = f_read_pdb_line(index, lines)
                if line[0:6] in set(['HETATM', 'ATOM  ']) and chainID == chain and position == pos:

                    ligand_name = line[17:20].strip()
                    atomName = line[12:16].strip()
                    altLoc = line[16]
                    atomSym = line[76:78].strip()

                    nex_ligand_name = lines[index + 1][17:20].strip()
                    next_chainID = lines[index + 1][21]
                    next_position = lines[index + 1][22:26].strip()
                    next_altLoc = lines[index + 1][16]

                    next_nex_ligand_name = lines[index + 2][17:20].strip()
                    next_next_chainID = lines[index + 2][21]
                    next_next_position = lines[index + 2][22:26].strip()
                    next_next_altLoc = lines[index + 2][16]

                    if (altLoc == ' ' or altLoc == 'A'):  # skip alternate position:
                        if ligand_name == target_ligand:
                            index_list.append(index)
                            data.append((atomSym,[float(i) for i in axes]))
                        else:
                            incomplete_indicator = True
                            break
                    end = f_molecule_not_end(index, lines, chain, pos)
                    if (next_position != position or (lines[index + 1][0:6] not in set(['HETATM', 'ATOM  ']))
                        or next_chainID != chainID) and ((next_next_position != position or next_next_chainID != chainID) or lines[index + 2][0:6] not in set(['HETATM', 'ATOM  '])):
                        name_list.popleft()
                        # print(name_list)
                        break
            if incomplete_indicator:
                break

            pos = str(int(pos) + 1)

        if len(name_list) != 0:
            # print(f'ligand {name} complex reading is failed')
            return 1
        else:
            return data

# lst = read_ligand_complex('./data/2w1u.pdb', 'NGA NAG', 'F', '1')

def label_atom(resname,atom):
    atom_label={
    'ARG':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':4,'NE':11,'CZ':1,'NH1':11,'NH2':11},
    'HIS':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':6,'CD2':6,'CE1':6,'ND1':8,'NE2':8},
    'LYS':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':4,'CE':4,'NZ':10},
    'ASP':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'OD1':15,'OD2':15},
    'GLU':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':2,'OE1':15,'OE2':15},
    'SER':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'OG':13},
    'THR':{'N':17,'CA':18,'C':19,'O':20,'CB':3,'OG1':13,'CG2':5},
    'ASN':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':1,'OD1':14,'ND2':9},
    'GLN':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':1,'OE1':14,'NE2':9},
    'CYS':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'SG':16},
    'GLY':{'N':17,'CA':18,'C':19,'O':20},
    'PRO':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':4},
    'ALA':{'N':17,'CA':18,'C':19,'O':20,'CB':5},
    'VAL':{'N':17,'CA':18,'C':19,'O':20,'CB':3,'CG1':5,'CG2':5},
    'ILE':{'N':17,'CA':18,'C':19,'O':20,'CB':3,'CG1':4,'CG2':5,'CD1':9},
    'LEU':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':3,'CD1':5,'CD2':5},
    'MET':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'SD':16,'CE':5},
    'PHE':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':6,'CD1':6,'CD2':6,'CE1':6,'CE2':6,'CZ':6},
    'TYR':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':6,'CD1':6,'CD2':6,'CE1':6,'CE2':6,'CZ':6,'OH':13},
    'TRP':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':6,'CD1':6,'CD2':6,'NE1':7,'CE2':6,'CE3':6,'CZ2':6,'CZ3':6,'CH2':6}
    }
    if atom == 'OXT':
        return 21   # define the extra oxygen atom OXT on the terminal carboxyl group as 21 instead of 27 (changed on March 19, 2018)
    else:
        return atom_label[resname][atom]


def flatten(t):
    return [item for sublist in t for item in sublist]



