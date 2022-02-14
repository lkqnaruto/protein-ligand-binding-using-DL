import os

import numpy as np
import pandas as pd
import sys
import json
from collections import deque, defaultdict


def list_files_recursive(input_path):
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
                    label = label_atom(resName, atomName)
                    if altLoc == ' ' or altLoc == 'A':  # skip alternate position
                        data.append([label,x,y,z])
    return data


def read_ligand(file_name, name):
    data = []
    with open(file_name) as f:
        for line in f:
            if line[0:6] == 'HETATM':
                atomName = line[12:16].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                altLoc = line[16]
                if altLoc == ' ' or altLoc == 'A':  # skip alternate position
                    if atomName == name:
                        data.append([22, x, y, z])
    return data


def read_ligand_simple(file_name, name, chain, pos):
    data = []
    with open(file_name) as f:
        for line in f:
            if line[0:6] == 'HETATM':
                atomName = line[12:16].strip()
                altLoc = line[16]
                if altLoc == ' ' or altLoc == 'A':  # skip alternate position

                    ligand_name = line[17:20].strip()
                    chainID = line[21]
                    position = line[22:26].strip()
                    atomSym = line[76:78].strip()
                    if chainID == chain and position == pos:
                        if ligand_name != name:
                            return 1
                        data.append(atomSym)
                # if atomSym == name:
                #     data.append([22, x, y, z])

        if data == []:
            return 2
        return (list(set(data)), data)

def read_ligand_complex(file_name, name, chain, pos):
    '''
    :param file_name:
    :param name: ligand name
    :param chain:
    :param pos:
    :return:
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
            # print(target_ligand, pos)
            assert len(lines[0]) == len(lines[1])
            for index, line in enumerate(lines):
                if line[0:6] in set(['HETATM', 'ATOM  ']) and line[21] == chain and line[22:26].strip() == pos:

                    ligand_name = line[17:20].strip()
                    atomName = line[12:16].strip()
                    chainID = line[21]
                    position = line[22:26].strip()
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
                            data.append(atomSym)
                        else:
                            incomplete_indicator = True
                            break

                    if (next_position != position or (lines[index + 1][0:6] not in set(['HETATM', 'ATOM  ']))
                        or next_chainID != chainID) and ((next_next_position != position or next_next_chainID != chainID) or lines[index + 2][0:6] not in set(['HETATM', 'ATOM  '])):
                        name_list.popleft()
                        # print(name_list)
                        break
            if incomplete_indicator:
                break
            # print(index_list)

            pos = str(int(pos) + 1)

        if len(name_list) != 0:
            # print(f'ligand {name} complex reading is failed')
            return 1
        else:
            return (list(set(data)), data)



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


if __name__=="__main__":

    target_ligands_file = './script/saved_ligands_new.txt'
    all_ligand_data_file = './script/ligand_dict_new.json'

    pdb_files_dir = './data/'
    pdb_files = os.listdir(pdb_files_dir)

    all_ligand_atom_type = defaultdict(list)
    with open(all_ligand_data_file, 'r') as f:
        all_ligand_data = json.load(f)

    with open(target_ligands_file, 'r') as file:

        for count, ligand in enumerate(file):
            # print(ligand)
            if len(ligand.strip().split(" "))==1:
                key_ligand = ligand.strip().strip('\"')
                pdb_info_list = all_ligand_data[key_ligand]
                # ligand_atom_type = None

                data_get_indicator = False
                for pdb_info in pdb_info_list:
                    chain_name, position, pdbid = pdb_info
                    if pdbid == '6G8H':
                        continue
                    # print(chain_name, position, pdbid)
                    target_pdb_file = pdb_files_dir + pdbid.lower() + '.pdb'
                    read_results = read_ligand_simple(target_pdb_file, key_ligand, chain_name, position)
                    if read_results not in [1,2]:
                        # print(read_results)
                        (ligand_atom_type, ligand_atom_all) = read_results
                        all_ligand_atom_type[key_ligand].append([ligand_atom_type, pdbid, ligand_atom_all])
                        data_get_indicator = True
                    else:
                        if read_results == 1:
                            print("ligand name in pdb is not consistent with MOAD")
                            # continue
                        else:
                            assert read_results == 2
                            print(f"reading {key_ligand}, but can not find it in PDB file. Current pdbid is {pdbid}")
                            # continue
                        # else:
                        #     read_results = 3
                        #     break
                # if read_results == 3:
                #     all_ligand_atom_type[key_ligand] = [ligand_atom_type, pdbid, ligand_atom_all]
                if not data_get_indicator :
                    print(f"current ligand is : {key_ligand}, and there is no pdb file contains its corresponding data")
                    continue
                # all_ligand_atom_type[key_ligand] = (ligand_atom_type, pdbid, ligand_atom_all)


            else:
                assert len(ligand.strip().split(" "))>1
                key_ligand_complex = ligand.strip().strip('\"')
                pdb_info_list = all_ligand_data[key_ligand_complex]

                for pdb_info in pdb_info_list:
                    chain_name, position, pdbid = pdb_info
                    # print(chain_name, position, pdbid)
                    target_pdb_file = pdb_files_dir + pdbid.lower() + '.pdb'
                    read_results = read_ligand_complex(target_pdb_file, key_ligand_complex, chain_name, position)
                    if read_results == 1:
                        print(f"ligand complex {key_ligand_complex} reading is not successful in PDB file {pdbid}.pdb")
                        continue
                    else:
                        (ligand_atom_type, ligand_atom_all) = read_results
                    all_ligand_atom_type[key_ligand].append([ligand_atom_type, pdbid, ligand_atom_all])

    with open('/Users/keqiaoli/Desktop/test_files/summary_atoms_all.json', 'w') as outputfile:
        outputfile.write(json.dumps(all_ligand_atom_type, indent = 4))

        print(all_ligand_atom_type)



























    # # target_ligands = list(pd.read_csv('all_valid_ligands.csv')['x'])
    # #
    # # input_paths = ['/Users/keqiaoli/Downloads/Selected_Bindingdata']
    # # file_list = list()
    # # for input_path in input_paths:
    # #     file_list.extend(list_files_recursive(input_path))
    # # # print(file_list)
    # # file_list.remove('/Users/keqiaoli/Downloads/Selected_Bindingdata/.DS_Store')
    # # file_list.remove('/Users/keqiaoli/Downloads/Selected_Bindingdata/.Rhistory')
    # #
    # # counter = 0
    # #
    # # file_list_small= file_list
    # # ligand_name_dict = {}
    # # ligand_files = {}
    # # # ['/Users/keqiaoli/Downloads/Selected_Bindingdata/Selected_Bindingdata-22/4duw.csv']
    # # for ligand in target_ligands:
    # #     for file in file_list:
    # #
    # #         if target_ligands == []:
    # #             break
    # #         key = file.split('.')[0][-4:]
    # #         print(key)
    # #         print(file)
    # #
    # #         df = pd.read_csv(file)
    # #
    # #         ligand_names = df[key.upper()]
    # #         ligand_names_list = [element.split(':')[0] for element in ligand_names]
    # #         ligand_names_list_expand = [ele.split(' ') for ele in ligand_names_list]
    # #         current_ligand_names_list_full = flatten(ligand_names_list_expand)
    # #
    # #         if ligand in current_ligand_names_list_full:
    # #             print("file founded")
    # #             # if key == '1h12':
    # #             #     continue
    # #             # if key == '4n2r':
    # #             #     continue
    # #             ligand_files[ligand] = key
    # #             break
    # #             # target_ligands.remove(ligand)
    # #
    # # # file_name = ligand_files['NAG']
    # # print(ligand_files)
    # # outputfile = open('list_files_for_downloading.txt',mode='w', encoding='utf-8')
    # # print(len([*ligand_files.keys()]))
    # # for ligand, file in ligand_files.items():
    # #     outputfile.write(file)
    # #     outputfile.write(',')
    #
    # # def read_ligand(file_name):
    #
    # from urllib.request import urlopen
    #
    # #
    #
    #
    #
    #
    # """
    # ligand_atoms_dict = {}
    # for ligand_name, file_name in ligand_files.items() :
    #     # print(file_name)
    #     # print(ligand_name)
    #     data = []
    #     with open(file_name + '.pdb') as f:
    #         for line in f:
    #             if line[0:6] == 'HETATM':
    #                 atomName = line[12:16].strip()
    #                 # print(atomName)
    #                 altLoc = line[16]
    #                 if altLoc == ' ' or altLoc == 'A':  # skip alternate position
    #                     # sys.exit()
    #                     if line[17:20] == ligand_name:
    #                         x = float(line[30:38].strip())
    #                         y = float(line[38:46].strip())
    #                         z = float(line[46:54].strip())
    #                         element = line[76:78].strip()
    #                         data_temp = []
    #                         # data_temp.append([22, x, y, z])
    #                         data_temp.append(element)
    #                         atomind = line[22:29].strip()
    #                         chainind = line[21]
    #                         # print(atomind)
    #                         # print(chainind)
    #                         # print(data_temp)
    #                         for item in f:
    #                             if item[0:6] == 'HETATM' and item[21] == chainind and item[16] == altLoc and item[
    #                                                                                                          17:20] == ligand_name:
    #                                 if item[22:29].strip() == atomind: #and item[12:16].strip() != 'S':
    #                                     x = float(item[30:38].strip())
    #                                     y = float(item[38:46].strip())
    #                                     z = float(item[46:54].strip())
    #                                     data_temp.append(item[76:78].strip())
    #                         data.append(set(data_temp))
    # #         # return data
    #     ligand_atoms_dict[ligand_name] = list(data[0])
    # print(ligand_atoms_dict)
    #
    # import csv
    # with open('test.csv', 'w') as file:
    #     for key in ligand_atoms_dict.keys():
    #         file.write("%s, %s\n" % (key, ligand_atoms_dict[key]))
    #     # print(data)
    # """
    #     # print(ligand_name)
    #     # print(file_name)
    #     print(data)
    #     length_dic[ligand_name] = len(data[0])
    #     print(length_dic)
    #
    #     import csv
    #     output_file = open("summary_atoms_number.csv", "w")
    #
    #     writer = csv.writer(output_file)
    #     for key, value in length_dic.items():
    #         writer.writerow([key, value])
    #
    #     output_file.close()


    #
    #


    # print(ligand_files)

    # print(len(ligand_files))



    # dates = pd.date_range('1/1/2000', periods=8)
    # df = pd.DataFrame(np.random.randn(8, 4), index = dates, columns = ['A', 'B', 'C', 'D'])
    # print(df)
    # print(type(df))
    # print(type(df['A']))
    # aa = list(df['A'])
    # print(len(aa))
    # print(aa)