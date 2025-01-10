import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

from utils.getModules import load_dataloaders, pIC50_to_IC50
from networks.chembert import ChemBERTpIC50Predictor


numfeatures = 18 # 분자, 원자 특성들
batch_size = 32

# device 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer and ChemBERT model
tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR')
bert_model = AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MTR')

# test_loader
_, _, test_loader = load_dataloaders(tokenizer, batch_size)

# Load the best model
model = ChemBERTpIC50Predictor(bert_model, numfeatures)
model.load_state_dict(torch.load('weights/best_model.pth'))
model.to(device)

# final predict
model.eval()
predictions = [] # 최종 예측들

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)

        outputs = model(input_ids, attention_mask, features)
        predictions.extend(outputs.squeeze().cpu().numpy())

# pIC50 to IC50
test_predictions = pIC50_to_IC50(np.array(predictions))

# 최종 예측 확인
print(f"Number of test predictions: {len(test_predictions)}")
print(f"Sample of predictions (IC50 values): {test_predictions[:5]}")

# submission 생성
submission = pd.read_csv('../data/sample_submission.csv')
submission['IC50_nM'] = test_predictions
submission.to_csv(f'submission/ChemBERTa-77M-MTR4.csv', index=False)
