# FILE CONTAINS ALL THE FUNCTIONS THAT ARE RELATED TO THE EXTRACTION AND INITIAL MANIPULATION OF ALL THE SCPDB

import os
import numpy as np
from biopandas.mol2 import PandasMol2
from openbabel import openbabel, OBMol, OBConversion
from Bio.PDB import PDBParser, PDBIO, Select
from common_funcs import *
# from funcs.common_funcs import *
import warnings
warnings.filterwarnings("ignore")

# Transform mol2 files contained in scPDB folders to .pdb
def mol2pdb(mol2path, pid, folder_pdb):
    # Open Babel conversion interface
    obconversion = OBConversion()
    # Turn off warning messages
    openbabel.obErrorLog.SetOutputLevel(0)
    # Set input and output formats
    obconversion.SetInAndOutFormats("mol2", "pdb") 
    # Create openbabel molecule instance
    obmol = OBMol()
    
    # Create folder where all .pdb generated files will be stored. 
    if os.path.isdir(folder_pdb) is not True:
        os.system(f'mkdir {folder_pdb}')
    
    # Some protein structures are repeated (there is more than 1 protein-ligand complex)
    counter = 0
    for directory in [f'{folder_pdb}/{pid}.pdb'] + [f'{folder_pdb}/{pid}_{i}.pdb' for i in range(2, 15)]:
        # Check if directory exists
        if os.path.exists(directory):
            counter += 1
                
    # Create the file if it has not been yet seen. 
    if counter == 0:
        obconversion.ReadFile(obmol, mol2path) 
        obconversion.WriteFile(obmol, f'{folder_pdb}/{pid}.pdb') 
        
    # Copy the already existing pdb and paste it with a new name (e.g., 1iki.pdb -> 1iki_2.pdb).
    # We copy because the protein is the same in different protein-ligand complexes. Difference will
    # be the possible final chain we might use depending on the position of the ligand. 
    else:
        os.system(f'cp {folder_pdb}/{pid}.pdb {folder_pdb}/{pid}_{counter+1}.pdb')
    
    return 0


# Return residues ids of those who belong to the first chain
def mol2ChainDetection(path):
    subst_id_chain = []
    main_chain_char = ''
    # We want to iterate only once through each file (saving comp. time)
    for p_idx, p in enumerate([path+'/ligand.mol2', path+'/protein.mol2']):
        subst_idx = np.Inf
        for idx, line in enumerate(open(p)):
            if '@<TRIPOS>SUBSTRUCTURE' in line:
                subst_idx = idx  
            # We are finished at this point
            if '@<TRIPOS>SET' in line:
                break
            # Get the rows that are between @SUBSTRUCTURE and @SET sections
            if idx > subst_idx:
                split_line = line.split()
                if p_idx == 0:
                    main_chain_char = split_line[5]
                    break
                else:
                    # Store substructure ID, name (just in case), and chain characters to which the ligand belongs.
                    subst_id_chain.append((int(split_line[0]), split_line[1], split_line[5]))
    
    # CASES IN WHICH LIGAND.MOL2 CHAIN != PROTEIN.MOL2
    chains = [subst_line[2] for subst_line in subst_id_chain]
    if main_chain_char not in set(chains):
        return None, None, None
    
    # Get the substructure that belong to the first chain
    filt_subst_ids = [subst[0] for subst in subst_id_chain if subst[2] is main_chain_char]

    # Load protein and ligand structures
    pmol = PandasMol2().read_mol2(path+'/protein.mol2')
    lmol = PandasMol2().read_mol2(path+'/ligand.mol2')
    
    # Filter by substructure name
    filt_df = pmol.df[pmol.df['subst_id'].isin(filt_subst_ids)]
    
    # Compute distance between selected chain/structure centroid and ligand 
    centroid_chain_ligand_dist = calculateDistance(getCentroid(torch.Tensor(filt_df[['x','y','z']].values)), 
                                          getCentroid(torch.Tensor(lmol.df[['x','y','z']].values)))
    
    # Calculate the distance for each atom_protein - atom_ligand pair. 
    chain_ligand_dist = 10
    loop_break = False
    for atom_prot in torch.Tensor(filt_df[['x','y','z']].values):
        for atom_lig in torch.Tensor(lmol.df[['x','y','z']].values):
            atom_dist = calculateDistance(atom_prot.unsqueeze(0), atom_lig.unsqueeze(0))
            if  atom_dist < chain_ligand_dist:
                chain_ligand_dist = atom_dist
                # If minimum atomic distance is less than 3 we assume chain selected is correct. 
                if chain_ligand_dist < 3:
                    loop_break = True
                    break 
        #if atom-pair distance is below the established threshold don't keep iterating.
        if loop_break is True:
            break
                    
    # Verify the substructures id are correct
    assert set(filt_df['subst_id']) == set(filt_subst_ids)

    return filt_subst_ids, chain_ligand_dist, centroid_chain_ligand_dist


# Remove heteroatoms and keep first chain 
def cleanPDB(path, filt_subst_ids):
    pdb_file = path
    # Create instance that will contain protein structure 
    pdb_struct = PDBParser().get_structure(pdb_file.split('.')[0], pdb_file)
    # Get all the chains of the structure  
    # chains = [chain.id for chain in model for model in pdb]
    io = PDBIO()
    io.set_structure(pdb_struct)
    # Keep first chain (There is only one chain after .mol2 openbabel conversion)
    io.save(pdb_file, chainSelect())
    # Remove heteroatoms
    io.save(pdb_file, NonHetSelect(filt_subst_ids))
    return 0


# Remove heteroatoms
class NonHetSelect(Select):
    def __init__(self, filt_subst_ids):
        super().__init__()
        self.ids = filt_subst_ids

    def accept_residue(self, residue):
        # remove heteroatoms and residues that were detected in other chains. 
        if residue.id[0] == " " and residue.id[1] in self.ids:
            return 1  
        else:
            return 0  
    
    
# Keep one single chain  
class chainSelect(Select):
    def accept_chain(self, chain):
        return 1 if chain.id[0] == 'A' else 0


# Extract cavity geometrical data from file. Returns coordinates and features of the cavity.
def getCavityInfo(path):
    # Create pmol object. 
    pmol = PandasMol2().read_mol2(path)
    # Get coords and atom types. 
    coords = pmol.df[['x', 'y', 'z']].values
    # atom_types = pmol.df['atom_name'].values.tolist()
    
    return torch.tensor(coords, dtype = torch.get_default_dtype())


# Calculate the volume of the mesh that each point cloud/cavity conforms. 
def getPocketVolume(cavity):
    return ss.ConvexHull(cavity).volume


# Extract SCPDB data.  
def getScData(paths, NUM_PATHS, folder_pdb):
    # Number of cavities to be extracted can be set (for now). 
    pdb_dict = {}
    # Counter that keeps track of the number 
    skip_count = 0
    
    # Create necessary structures
    sc_cavities, sc_volume, points_per_pocket_sc = [], [], []
    residues_ids = []
    # Structures where pockets were not found due fpocket settings
    non_pck_list = ['1got', '1kgi', '1mz9', '4gbn', '5bqq', '2ws7', '4bb2', '1tz8', '4gbi', 
        '5ahw', '4aiy', '3aiy', '2jj1', '1fbm', '2krd', '4gbc', '2w44', '2aiy', '4e7v', '1pkv', '1ie4','5aiy'
        , '2h6i', '2eph']

    # Iterate through the paths. 
    for idx, protein_path in enumerate(paths[:NUM_PATHS]):
        # Get structures ID
        pid = protein_path.split('/')[-1].split('_')[0]
        
        # ----------REMOVE SPECIAL CASES (NO POCKETS FOUND) ------------------
        if pid in non_pck_list:
            skip_count += 1
            # print(f'{pid} dismissed manually.')
            continue
            
        # Get the residue ids of the first chain 
        mol2_residues_ids, cldist, ccldist = mol2ChainDetection(protein_path)
        
        # If the ligand does not reference any chain of the protein.mol2, we dismiss it.
        if mol2_residues_ids is None: 
            # print(f'{pid} dismissed. Wrong chain')
            skip_count += 1
            continue 
        
        # If centroids distance is too big, we need to make sure atom-atom distance is low to make sure 
        # ligand is in the stated chain
        if cldist > 3:
            # print(f'{pid} dismissed. Atom pair distance')
            skip_count += 1
            continue 
        
        # Distance between selected chain centroid and ligands
        #centroid_chain_lig_dist.append(ccldist)
        
        # Minimum atom_protein - atom_ligand distance in each protein-ligand complex
        # chain_lig_dist.append(cldist)
        
        # Collect residues from selected chain for posterior cleaning step in remaining .pdb file. 
        residues_ids.append(mol2_residues_ids)
        # Get coordinates, atom types, and ID
        coords = getCavityInfo(protein_path+'/cavity6.mol2')
        # Append set of coordinates.
        sc_cavities.append(coords)
        # Append number of coordinates/points cavity has. 
        points_per_pocket_sc.append(len(coords))
        # Calculate the volume
        sc_volume.append(getPocketVolume(coords))
        
        # Generate dictionary where key is the pdb ID and value the reading order.
        if pdb_dict.get(pid) is None:
            pdb_dict[pid] = [idx-skip_count]
        else:
            pdb_dict[pid].append(idx-skip_count)
        
        # Mol2 conversion to pdb. Folder is created here.
        mol2pdb(protein_path+'/protein.mol2', pid, folder_pdb)

    print(f'{skip_count} cavities dismissed due to not meeting requirements.')
    print('Cleaning PDB files...')
    # CLEAN .PDB FILES 
    for idx, path in enumerate(os.walk(folder_pdb)):
        for pid in path[2]:
            cleanPDB(path[0]+'/'+pid, residues_ids[pdb_dict[pid.replace('.pdb','').split('_')[0]][0]])

    
    return sc_cavities, points_per_pocket_sc, pdb_dict, sc_volume