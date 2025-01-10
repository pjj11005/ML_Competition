import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModel

from utils.tools import evaluate
from utils.getModules import load_dataloaders
from networks.chembert import ChemBERTpIC50Predictor

# Hparam 설정
batch_size = 32
numfeatures = 18 # 분자, 원자 특성들
num_epochs = 50

# device 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer and ChemBERT model
tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR')
bert_model = AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MTR')

## 데이터 처리하는 과정 (Molecule data)
train_loader, val_loader, _ = load_dataloaders(tokenizer, batch_size)

# 모델 객체를 생성 (설계도 + Hparam)
model = ChemBERTpIC50Predictor(bert_model, numfeatures).to(device)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, features)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss (pIC50 MSE): {avg_loss:.4f}')

    avg_val_loss = evaluate(model, val_loader, device, criterion)
    print(f'Validation Loss (pIC50 MSE): {avg_val_loss:.4f}')

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), 'weights/best_model.pth')
    else:
        counter += 1

    if counter >= patience:
        print(f'Early stopping after {epoch+1} epochs')
        break
