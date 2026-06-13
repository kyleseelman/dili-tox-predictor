"""
Molecular featurization for DILI prediction.

Mirrors Axiom Bio's approach of combining:
  - Morgan fingerprints (circular fingerprints capturing local chemistry)
  - Physicochemical descriptors (lipophilicity, TPSA, HBD/HBA, rotatable bonds)
  - ADME-relevant properties computed from structure
  - Graph representations for GNN models
"""

from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem,
    Descriptors,
    rdMolDescriptors,
    rdFingerprintGenerator,
    Lipinski,
    Crippen,
    MolSurf,
    GraphDescriptors,
)
from rdkit.DataStructs import ConvertToNumpyArray

RDLogger.logger().setLevel(RDLogger.ERROR)


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    return mol


_FP_GENERATORS: dict[tuple[int, int], object] = {}


def morgan_fingerprint(mol: Chem.Mol, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    key = (radius, nbits)
    if key not in _FP_GENERATORS:
        _FP_GENERATORS[key] = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=nbits
        )
    gen = _FP_GENERATORS[key]
    fp = gen.GetFingerprintAsNumPy(mol)
    return fp.astype(np.float32)


DESCRIPTOR_NAMES = [
    "MolWt",
    "LogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumHeavyAtoms",
    "FractionCSP3",
    "MolLogP",
    "MolMR",
    "LabuteASA",
    "BalabanJ",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "RingCount",
    "HeavyAtomMolWt",
    "NumValenceElectrons",
    "MaxPartialCharge",
    "MinPartialCharge",
]


def physicochemical_descriptors(mol: Chem.Mol) -> dict:
    """Compute physicochemical descriptors relevant to hepatotoxicity risk."""
    desc = {}
    desc["MolWt"] = Descriptors.MolWt(mol)
    desc["LogP"] = Crippen.MolLogP(mol)
    desc["TPSA"] = Descriptors.TPSA(mol)
    desc["NumHDonors"] = Lipinski.NumHDonors(mol)
    desc["NumHAcceptors"] = Lipinski.NumHAcceptors(mol)
    desc["NumRotatableBonds"] = Lipinski.NumRotatableBonds(mol)
    desc["NumAromaticRings"] = Descriptors.NumAromaticRings(mol)
    desc["NumHeavyAtoms"] = Descriptors.HeavyAtomCount(mol)
    desc["FractionCSP3"] = Descriptors.FractionCSP3(mol)
    desc["MolLogP"] = Crippen.MolLogP(mol)
    desc["MolMR"] = Crippen.MolMR(mol)
    desc["LabuteASA"] = MolSurf.LabuteASA(mol)
    try:
        desc["BalabanJ"] = GraphDescriptors.BalabanJ(mol)
    except Exception:
        desc["BalabanJ"] = 0.0
    desc["NumAliphaticRings"] = Descriptors.NumAliphaticRings(mol)
    desc["NumSaturatedRings"] = Descriptors.NumSaturatedRings(mol)
    desc["RingCount"] = Descriptors.RingCount(mol)
    desc["HeavyAtomMolWt"] = Descriptors.HeavyAtomMolWt(mol)
    desc["NumValenceElectrons"] = Descriptors.NumValenceElectrons(mol)
    desc["MaxPartialCharge"] = Descriptors.MaxPartialCharge(mol)
    desc["MinPartialCharge"] = Descriptors.MinPartialCharge(mol)
    return desc


def lipinski_alerts(mol: Chem.Mol) -> dict:
    """Flag Lipinski Rule-of-5 violations, relevant to oral drug hepatotoxicity."""
    return {
        "Ro5_MolWt_violation": int(Descriptors.MolWt(mol) > 500),
        "Ro5_LogP_violation": int(Crippen.MolLogP(mol) > 5),
        "Ro5_HBD_violation": int(Lipinski.NumHDonors(mol) > 5),
        "Ro5_HBA_violation": int(Lipinski.NumHAcceptors(mol) > 10),
        "Ro5_total_violations": sum([
            Descriptors.MolWt(mol) > 500,
            Crippen.MolLogP(mol) > 5,
            Lipinski.NumHDonors(mol) > 5,
            Lipinski.NumHAcceptors(mol) > 10,
        ]),
    }


def featurize_single(smiles: str, fp_radius: int = 2, fp_nbits: int = 2048) -> Optional[dict]:
    """Full featurization of a single molecule."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    fp = morgan_fingerprint(mol, fp_radius, fp_nbits)
    desc = physicochemical_descriptors(mol)
    alerts = lipinski_alerts(mol)

    return {
        "fingerprint": fp,
        "descriptors": desc,
        "alerts": alerts,
    }


def featurize_dataset(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    fp_radius: int = 2,
    fp_nbits: int = 2048,
) -> tuple[np.ndarray, np.ndarray, list[str], list[bool]]:
    """
    Featurize an entire dataset.

    Returns:
        fp_matrix: (N, fp_nbits) Morgan fingerprint matrix
        desc_matrix: (N, n_desc) physicochemical descriptor matrix
        desc_names: list of descriptor names
        valid_mask: boolean mask of successfully featurized rows
    """
    fps = []
    descs = []
    valid = []

    for smi in df[smiles_col]:
        feat = featurize_single(smi, fp_radius, fp_nbits)
        if feat is not None:
            fps.append(feat["fingerprint"])
            d = {**feat["descriptors"], **feat["alerts"]}
            descs.append(d)
            valid.append(True)
        else:
            valid.append(False)

    fp_matrix = np.stack(fps)
    desc_df = pd.DataFrame(descs)
    desc_names = list(desc_df.columns)
    desc_matrix = desc_df.values.astype(np.float32)

    nan_mask = np.isnan(desc_matrix)
    if nan_mask.any():
        col_means = np.nanmean(desc_matrix, axis=0)
        for j in range(desc_matrix.shape[1]):
            desc_matrix[nan_mask[:, j], j] = col_means[j]

    return fp_matrix, desc_matrix, desc_names, valid


# --- Graph featurization for GNNs ---

ATOM_FEATURES = {
    "atomic_num": list(range(1, 120)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "num_hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
}


def one_hot(value, choices: list) -> list[int]:
    enc = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
        enc[idx] = 1
    except ValueError:
        enc[-1] = 1
    return enc


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
        + one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])
        + one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
        + one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
        + [int(atom.GetIsAromatic())]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        one_hot(bond.GetBondType(), BOND_FEATURES["bond_type"])
        + [int(bond.GetIsConjugated())]
        + [int(bond.IsInRing())]
    )


def mol_to_graph(smiles: str) -> Optional[dict]:
    """Convert SMILES to a graph representation for GNNs."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom))

    edge_index = []
    edge_feats = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_feats.extend([bf, bf])

    if len(edge_index) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feats = np.zeros((0, 7), dtype=np.float32)
    else:
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_feats = np.array(edge_feats, dtype=np.float32)

    return {
        "node_feats": np.array(node_feats, dtype=np.float32),
        "edge_index": edge_index,
        "edge_feats": edge_feats,
        "num_nodes": len(node_feats),
    }
