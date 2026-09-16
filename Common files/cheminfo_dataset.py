import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from rdkit import Chem

"""Import PolymerSmilesTokenizer from PolymerSmilesTokenization.py"""
from PolymerSmilesTokenization import PolymerSmilesTokenizer

"""Downstream Dataset"""
class Downstream_Dataset(Dataset):
    def __init__(self, dataset, tokenizer, max_token_len):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_token_len = max_token_len

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, i):
        data_row = self.dataset.iloc[i]
        seq = data_row.iloc[0]
        new_feats = data_row.iloc[1:-1].astype(float).values
        prop = data_row.iloc[-1]

        encoding = self.tokenizer(
            str(seq),
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            prop=prop,
            new_feats = torch.tensor(new_feats)
        )


class DataAugmentation:
    def __init__(self, aug_indicator):
        super(DataAugmentation, self).__init__()
        self.aug_indicator = aug_indicator

    """Rotate atoms to generate more SMILES"""
    def rotate_atoms(self, li, x):
        return (li[x % len(li):] + li[:x % len(li)])

    """Generate SMILES"""
    def generate_smiles(self, smiles):
        smiles_list = []
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            mol = None
        if mol != None:
            n_atoms = mol.GetNumAtoms()
            n_atoms_list = [nat for nat in range(n_atoms)]
            if n_atoms != 0:
                for iatoms in range(n_atoms):
                    n_atoms_list_tmp = self.rotate_atoms(n_atoms_list, iatoms)  # rotate atoms' index
                    nmol = Chem.RenumberAtoms(mol, n_atoms_list_tmp)  # renumber atoms in mol
                    try:
                        smiles = Chem.MolToSmiles(nmol,
                                                  isomericSmiles=True,  # keep isomerism
                                                  kekuleSmiles=False,  # kekulize or not
                                                  rootedAtAtom=-1,  # default
                                                  canonical=False,  # canonicalize or not
                                                  allBondsExplicit=False,  #
                                                  allHsExplicit=False)  #
                    except:
                        smiles = 'None'
                    smiles_list.append(smiles)
            else:
                smiles = 'None'
                smiles_list.append(smiles)
        else:
            try:
                smiles = Chem.MolToSmiles(mol,
                                          isomericSmiles=True,  # keep isomerism
                                          kekuleSmiles=False,  # kekulize or not
                                          rootedAtAtom=-1,  # default
                                          canonical=False,  # canonicalize or not
                                          allBondsExplicit=False,  #
                                          allHsExplicit=False)  #
            except:
                smiles = 'None'
            smiles_list.append(smiles)
        smiles_array = pd.DataFrame(smiles_list).drop_duplicates().values
        # """
        if self.aug_indicator is not None:
            smiles_aug = smiles_array[1:, :]
            np.random.shuffle(smiles_aug)
            smiles_array = np.vstack((smiles_array[0, :], smiles_aug[:self.aug_indicator-1, :]))
        return smiles_array

    """SMILES Augmentation"""
    def smiles_augmentation(self, df):
        column_list = df.columns
        data_aug = np.zeros((1, df.shape[1]))
        for i in range(df.shape[0]):
            smiles = df.iloc[i, 0]
            new_feats_and_props = df.iloc[i, 1:]
            smiles_array = self.generate_smiles(smiles)
            if 'None' not in smiles_array:
                new_feats_and_props = np.tile(new_feats_and_props, (len(smiles_array), 1))
                data_new = np.hstack((smiles_array, new_feats_and_props))
                data_aug = np.vstack((data_aug, data_new))
        df_aug = pd.DataFrame(data_aug[1:, :], columns=column_list)
        return df_aug