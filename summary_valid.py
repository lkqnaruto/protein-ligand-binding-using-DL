## 2. Keep VALID ligands (no missing atoms aka the LONGEST)
# Dictionary to decide valid length
list(all_ligand_atom_type.items())[0]
ligand_name = [k for k, v in all_ligand_length.items()]
for k,v in all_ligand_length.items():
    v = set([(i,j) for i,j in v])
    v1 = [pdbid for (pdbid,length) in v]
    v2 = [length for (pdbid, length) in v]
    length = np.max(v2)
    valid_pdbid = [i for (i,j) in zip(v1,idx) if j == True]







# Read protein env
