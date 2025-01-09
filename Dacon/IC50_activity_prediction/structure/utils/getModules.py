import pandas as pd
from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from rdkit.Chem import Descriptors


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


# 데이터 로드
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

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
