import torch
from torch_geometric.data import Data
from rdkit import Chem
from Bio.PDB import PDBParser

from rdkit import Chem
import torch
from torch_geometric.data import Data

def mol_to_graph(smiles):
    """
    Convert a ligand SMILES string into a PyTorch Geometric graph.
    Workflow reference: Section 5.2 Ligand Graph Construction
    """

    # Parse molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # -------------------------
    # Node features (atoms)
    # -------------------------
    # Features per atom:
    # [atomic_number, degree, formal_charge, aromaticity]
    x = []

    for atom in mol.GetAtoms():
        x.append([
            atom.GetAtomicNum(),          # Atomic number
            atom.GetTotalDegree(),        # Degree
            atom.GetFormalCharge(),       # Formal charge
            int(atom.GetIsAromatic())     # Aromaticity (0/1)
        ])

    x = torch.tensor(x, dtype=torch.float)

    # -------------------------
    # Edge index & edge features (bonds)
    # -------------------------
    edge_index = []
    edge_attr = []

    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4
    }

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_type = bond_type_map[bond.GetBondType()]

        # Undirected graph → add both directions
        edge_index.append([i, j])
        edge_index.append([j, i])

        edge_attr.append([bond_type])
        edge_attr.append([bond_type])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # -------------------------
    # PyG Data object
    # -------------------------
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    return data


# Amino acid mapping
AA_LIST = [
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
]
aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}

# Biochemical properties
# Kyte–Doolittle hydropathy index
hydrophobicity = {
    'ALA':1.8,'ARG':-4.5,'ASN':-3.5,'ASP':-3.5,'CYS':2.5,
    'GLN':-3.5,'GLU':-3.5,'GLY':-0.4,'HIS':-3.2,'ILE':4.5,
    'LEU':3.8,'LYS':-3.9,'MET':1.9,'PHE':2.8,'PRO':-1.6,
    'SER':-0.8,'THR':-0.7,'TRP':-0.9,'TYR':-1.3,'VAL':4.2
}

charged = {'ARG':1,'LYS':1,'HIS':1,'ASP':-1,'GLU':-1}
polar = {'ASN','ASP','GLN','GLU','HIS','SER','THR','TYR'}

import Bio.PDB as PDB # Added this import
import numpy as np # Import numpy
def protein_to_graph(pdb_file, cutoff=10.0):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    coords = []
    features = []

    for model in structure:
        for chain in model:
            for res in chain:
                if PDB.is_aa(res) and 'CA' in res:
                    resname = res.get_resname()
                    if resname not in aa_to_idx:
                        continue

                    # One-hot encoding
                    one_hot = [0]*20
                    one_hot[aa_to_idx[resname]] = 1

                    # Biochemical features
                    hydro = hydrophobicity[resname]
                    charge = charged.get(resname, 0)
                    pol = 1 if resname in polar else 0

                    features.append(one_hot + [hydro, charge, pol])
                    coords.append(res['CA'].coord)

    coords = np.array(coords)
    x = torch.tensor(features, dtype=torch.float)

    # Edge construction
    edge_index = []
    N = len(coords)
    for i in range(N):
        for j in range(N):
            if i != j:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < cutoff:
                    edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    return Data(x=x, edge_index=edge_index)