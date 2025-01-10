import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from rdkit import Chem
from rdkit.Chem import Descriptors

from dataset.moleculedataset import MoleculeDataset


# pIC50 -> IC50
def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)


# 분자 특성
def calculate_molecular_descriptors(mol):
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
        "NumRings": Descriptors.RingCount(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "TopologicalPolarSurfaceArea": Descriptors.TPSA(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
    }


# 원자 특성
def calculate_atomic_features(mol):
    atom_features = {}
    for idx, atom in enumerate(mol.GetAtoms()):
        features = {
            "AtomicNum": atom.GetAtomicNum(),
            "Degree": atom.GetDegree(),
            "FormalCharge": atom.GetFormalCharge(),
            "NumRadicalElectrons": atom.GetNumRadicalElectrons(),
            "Hybridization": int(atom.GetHybridization()),
            "IsAromatic": int(atom.GetIsAromatic()),
            "Mass": atom.GetMass(),
            "TotalValence": atom.GetTotalValence(),
        }
        atom_features.update(features)
    return atom_features


# 특성 추출
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        molecular_descriptors = calculate_molecular_descriptors(mol)
        atomic_features = calculate_atomic_features(mol)
        return {**molecular_descriptors, **atomic_features}
    else:
        return None


# 전처리된 train, test 반환
def load_train_test():
    # 데이터 로드
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')

    # feature + target, X_test
    X_train = train_data["Smiles"]
    y_train = train_data["pIC50"]
    X_test = test_data["Smiles"]

    # 특성 추출
    X_train_features = X_train.apply(extract_features)
    X_test_features = X_test.apply(extract_features)

    # 결측치 제거
    X_train_features = X_train_features[X_train_features.notna()]
    y_train = y_train[X_train_features.index]
    X_test_features = X_test_features[X_test_features.notna()]

    # 데이터 프레임으로 변환
    X_train_features = pd.DataFrame(X_train_features.to_list())
    X_test_features = pd.DataFrame(X_test_features.to_list())

    # 수치형 스케일링
    scaler = StandardScaler()
    numerical_columns = X_train_features.columns[X_train_features.dtypes != "object"]
    X_train_features[numerical_columns] = scaler.fit_transform(X_train_features[numerical_columns])
    X_test_features[numerical_columns] = scaler.transform(X_test_features[numerical_columns])

    return X_train, X_test, X_train_features, X_test_features, y_train


# dataset 
def load_datasets(X_train, X_test, X_train_features, X_test_features, y_train,  tokenizer):
    train_dataset = MoleculeDataset(X_train, X_train_features, y_train, tokenizer)
    test_dataset = MoleculeDataset(X_test, X_test_features, tokenizer=tokenizer)

    return train_dataset, test_dataset


# dataloader
def load_dataloaders(tokenizer, batch_size):
    X_train, X_test, X_train_features, X_test_features, y_train = load_train_test()
    train_dataset, test_dataset = load_datasets(X_train, X_test, X_train_features, X_test_features, y_train,  tokenizer)

    # train data -> train and validation sets
    train_idx, val_idx = train_test_split(range(len(train_dataset)), test_size=0.2, random_state=42)
    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)

    # data loaders 생성 (train, valid, test)
    train_loader = DataLoader(train_dataset, batch_size, sampler=train_subsampler)
    val_loader = DataLoader(train_dataset, batch_size, sampler=val_subsampler)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
