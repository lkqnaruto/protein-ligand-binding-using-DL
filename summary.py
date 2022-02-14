from script.utils import *
import os
os.getcwd()

target_ligands_file = './script/saved_ligands_new.txt'
all_ligand_data_file = './script/ligand_dict_new.json'

pdb_files_dir = './data/'
pdb_files = os.listdir(pdb_files_dir)

all_ligand_atom_type = defaultdict(list)
with open(all_ligand_data_file, 'r') as f:
    all_ligand_data = json.load(f)

with open(target_ligands_file, 'r') as file:
    for count, ligand in enumerate(file):

        # Read ligand
        if len(ligand.strip().split(" "))==1:
            ligand_name = ligand.strip().strip('\"')
            pdb_info_list = all_ligand_data[ligand_name]
            data_get_indicator = False
            for pdb_info in pdb_info_list:
                chain_name, position, pdbid = pdb_info
                # if pdbid == '6G8H':
                #     continue
                target_pdb_file = pdb_files_dir + pdbid.lower() + '.pdb'
                try:
                    read_results = read_ligand_simple(target_pdb_file, ligand_name, chain_name, position)
                    if read_results not in [1,2]:
                        ligand_atom = [i for (i,j) in read_results]
                        axes = [j for (i,j) in read_results]
                        all_ligand_atom_type[ligand_name].append([pdbid, ligand_atom, axes])

                        data_get_indicator = True
                except:
                    print(pdbid)

                try:
                    read_results = read_ligand_complex(target_pdb_file, key_ligand_complex, chain_name, position)
                    if read_results != 1:
                        ligand_atom = [i for (i, j) in read_results]
                        axes = [j for (i, j) in read_results]
                        all_ligand_atom_type[ligand_name].append([pdbid, ligand_atom, axes])
                except:
                    print(pdbid)
        print(all_ligand_atom_type)

        # Read protein env



with open('./ligand_atom_axes.json', 'w') as outputfile:
    outputfile.write(json.dumps(all_ligand_atom_type, indent = 4))

    print(all_ligand_atom_type)

