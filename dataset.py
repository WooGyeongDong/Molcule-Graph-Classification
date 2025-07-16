import tempfile
import pandas as pd
import torch

from torch_geometric.datasets import TUDataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles

from ogb.graphproppred import PygGraphPropPredDataset

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def get_random_split(length):
    split_idx = {}
    ratio = round(length*0.1)
    train_idx = torch.randperm(length)
    split_idx["test"] = train_idx[:ratio]
    split_idx["valid"] = train_idx[ratio:ratio*2]
    split_idx["train"] = train_idx[ratio*2:]
    return split_idx

def get_MUTAG_smiles():
    node_label_map = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
    edge_label_map = {0: Chem.BondType.AROMATIC, 1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    dataset = TUDataset(root='./dataset', name='MUTAG')
    smiles_list = []
    for idx, data in enumerate(dataset):

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        node_labels = data.x.squeeze().tolist()

        if (idx in [82,187]): edge_attr[22] = torch.tensor([0.,1.,0.,0.])

        mol = Chem.RWMol()
        
        for node_label in node_labels:
            atom = Chem.Atom(node_label_map[node_label.index(1.0)])
            mol.AddAtom(atom)

        for k, (i, j) in enumerate(edge_index.t().tolist()):
            bond_type = edge_label_map[torch.argmax(edge_attr[k]).item()]
            try: mol.AddBond(int(i), int(j), bond_type)
            except: pass
            atom_i = mol.GetAtomWithIdx(int(i))
            atom_j = mol.GetAtomWithIdx(int(j))
            
            if bond_type == Chem.BondType.SINGLE:
                if (atom_i.GetSymbol() == 'N' and atom_j.GetSymbol() == 'O'):
                    atom_i.SetFormalCharge(1)
                    atom_j.SetFormalCharge(-1)
                elif (atom_i.GetSymbol() == 'O' and atom_j.GetSymbol() == 'N'):
                    atom_i.SetFormalCharge(-1)
                    atom_j.SetFormalCharge(1)
        
        if (idx in [13,41,88,119,137,177]): mol.GetAtomWithIdx(1).SetFormalCharge(1)
        if (idx==149): mol.GetAtomWithIdx(5).SetFormalCharge(1)
                    
        AllChem.SanitizeMol(mol)  
        smiles = Chem.MolToSmiles(mol)
        smiles_list.append(smiles)
    with open('./dataset/MUTAG/raw/MUTAG_smiles.txt', '+w') as f:
        f.write('\n'.join(smiles_list))

def get_MUTAG():
    dataset = TUDataset(root='./dataset', name='MUTAG')
    smiles_list = []
    try:
        with open('./dataset/MUTAG/raw/MUTAG_smiles.txt', 'r') as f:
            for line in f:
                smiles_list.append(line.strip())
    except:
        get_MUTAG_smiles()
        with open('./dataset/MUTAG/raw/MUTAG_smiles.txt', 'r') as f:
            for line in f:
                smiles_list.append(line.strip())

    datalist = []
    for idx, data in enumerate(dataset):
        smiles_data = from_smiles(smiles_list[idx])
        smiles_data.y = data.y.squeeze()
        
        datalist.append(smiles_data)
          
    return MoleculeDataset('./', datalist), get_random_split(len(datalist))

def get_MolculeNetData(name, target_col=None):
    name = name.lower()
    dataset = PygGraphPropPredDataset(name='ogbg-mol'+name)
    smiles = pd.read_csv(f'./dataset/ogbg_mol{name}/mapping/mol.csv.gz', compression = 'gzip')
    smiles = smiles['smiles'].to_list()

    datalist = []
    for idx, data in enumerate(dataset):
        data.smiles = smiles[idx]
        if target_col is None:
            data.y = data.y.squeeze()
        else:
            data.y = data.y.squeeze()[target_col]
        
        datalist.append(data)
   
    return MoleculeDataset('./', datalist), dataset.get_idx_split()

# %%
class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        self._temp_dir = tempfile.TemporaryDirectory()
        super().__init__(self._temp_dir.name, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        self.save(self.data_list, self.processed_paths[0])