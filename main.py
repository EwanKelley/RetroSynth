from rdkit import Chem
import pandas
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.nn import NNConv, global_mean_pool

ATOM_TYPES = [6, 7, 8, 9, 15, 16, 17, 35, 53]

class ReactionCenterGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, n_layers=3):
        super().__init__()

        self.convs = nn.ModuleList()
        in_dim = node_dim
        for _ in range(n_layers):
            edge_net = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_dim * hidden_dim)
            )
            self.convs.append(NNConv(in_dim, hidden_dim, edge_net, aggr='mean'))
            in_dim = hidden_dim
            
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x)
        
        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        
        out = self.edge_predictor(edge_repr).squeeze(-1)
        return out

def result_to_pyg(result: dict) -> Data:
    graph = result['product_graph']
    bond_labels = result['bond_labels']
    mol = Chem.MolFromSmiles(result['product_smi'])
    
    x = torch.tensor(graph['node_feats'], dtype=torch.float)
    edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
    edge_attr = torch.tensor(graph['edge_feats'], dtype=torch.float)
    
    labels = []

    for bond in mol.GetBonds():
        label = bond_labels[bond.GetIdx()]
        labels += [label, label]
    edge_y = torch.tensor(labels, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_y=edge_y)

def atom_features(atom):
    one_hot = [int(atom.GetAtomicNum() == t) for t in ATOM_TYPES]
    other = [
        atom.GetDegree(),
        atom.GetTotalValence(),
        int(atom.GetIsAromatic()),
        atom.GetFormalCharge(),
        atom.GetNumImplicitHs(),
        atom.GetAtomMapNum(),
    ]
    return one_hot + other

def get_edge_index(mol):
    src, dst = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    return [src, dst]

def bond_features(bond):
    bt = bond.GetBondTypeAsDouble()
    return [
        int(bt == 1.0),
        int(bt == 1.5),
        int(bt == 2.0),
        int(bt == 3.0),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]

def mol_to_graph(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_feats = [atom_features(a) for a in mol.GetAtoms()]

    src, dst = [], []
    edge_feats = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        edge_feats += [bf, bf]
        
    atom_map_nums = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    
    return {
        'node_feats': np.array(node_feats, dtype=np.float32),
        'edge_index': np.array([src, dst], dtype=np.int64),
        'edge_feats': np.array(edge_feats, dtype=np.float32),
        'atom_map_nums': atom_map_nums,
    } 

def get_reaction_center_labels(rxn_smiles: str) -> dict | None:
    reactants_smi, _, products_smi = rxn_smiles.split('>')

    product_mol = Chem.MolFromSmiles(products_smi)
    
    reactant_mols = [Chem.MolFromSmiles(s) for s in reactants_smi.split('.')]

    if product_mol is None or any(m is None for m in reactant_mols):
        return None

    reactant_bonds = {}
    for mol in reactant_mols:
        for bond in mol.GetBonds():
            map_i = bond.GetBeginAtom().GetAtomMapNum()
            map_j = bond.GetEndAtom().GetAtomMapNum()
            if map_i == 0 or map_j == 0:
                continue
            key = frozenset([map_i, map_j])
            reactant_bonds[key] = bond.GetBondTypeAsDouble()
            
    bond_labels = {}
    for bond in product_mol.GetBonds():
        map_i = bond.GetBeginAtom().GetAtomMapNum()
        map_j = bond.GetEndAtom().GetAtomMapNum()
        key = frozenset([map_i, map_j])

        product_bt = bond.GetBondTypeAsDouble()
        reactant_bt = reactant_bonds.get(key, None)

        if reactant_bt is None:
            label = 1
        elif reactant_bt != product_bt:
            label = 1
        else:
            label = 0
        bond_labels[bond.GetIdx()] = label
        
    return {
        'product_graph': mol_to_graph(products_smi),
        'product_smi': products_smi,
        'bond_labels': bond_labels,
        'rxn_class': None
    }

def labels_to_edge_label_vector(product_mol, bond_labels: dict) -> np.array:
    labels = []
    for bond in product_mol.GetBonds():
        label = bond_labels[bond.GetIdx()]
        labels += [label, label]
    return np.array(labels, dtype=np.float32)

train_df = pandas.read_csv('raw_train.csv')
val_df = pandas.read_csv('raw_val.csv')
test_df = pandas.read_csv('raw_test.csv')


print(len(train_df), len(val_df), len(test_df))

train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

df = pandas.concat([train_df, val_df, test_df], ignore_index=True)
print(f"Total: {len(df)}")

processed = {'train': [], 'val': [], 'test': []}
failed = 0

for _, row in df.iterrows():
    result = get_reaction_center_labels(row['reactants>reagents>production'])
    if result is None:
        failed += 1
        continue
    result['rxn_class'] = row['class']
    processed[row['split']].append(result)

print(f"Train: {len(processed['train'])}")
print(f"Val:   {len(processed['val'])}")
print(f"Test:  {len(processed['test'])}")
print(f"Failed: {failed}")

train_data = [result_to_pyg(r) for r in processed['train']]
val_data = [result_to_pyg(r) for r in processed['val']]
test_data = [result_to_pyg(r) for r in processed['test']]

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


d = train_data[0]
print(d.x.shape)
print(d.edge_index.shape)
print(d.edge_attr.shape)
print(d.edge_y.shape)

node_dim = train_data[0].x.shape[1]
edge_dim = train_data[0].edge_attr.shape[1]

model = ReactionCenterGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=64, n_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

pos_weight = torch.tensor([10.0])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def train_epoch(loader):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.edge_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            loss = criterion(logits, batch.edge_y)
            total_loss += loss.item()
    return total_loss / len(loader)

def topk_accuracy(loader, k=1):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            logits = model(batch)

            ptr = batch.ptr

            for i in range(batch.num_graphs):
                src = batch.edge_index[0]
                node_start = ptr[i].item()
                node_end = ptr[i+1].item()
                mask = (src >= node_start) & (src < node_end)
                
                graph_logits = logits[mask]
                graph_labels = batch.edge_y[mask]

                graph_logits = graph_logits[::2]
                graph_labels = graph_labels[::2]

                topk_indices = torch.topk(graph_logits, k=min(k, len(graph_logits))).indices

                true_centers = (graph_labels == 1).nonzero(as_tuple=True)[0]
                hit = any(idx in topk_indices for idx in true_centers)

                correct += int(hit)
                total += 1
    return correct / total

for epoch in range(30):
    train_loss = train_epoch(train_loader)
    val_loss = eval_epoch(val_loader)
    val_top1 = topk_accuracy(val_loader, k=1)
    val_top3 = topk_accuracy(val_loader, k=3)
    print(f"Epoch {epoch+1 :02d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Top-1: {val_top1:.3f} | Top-3: {val_top3:.3f}")