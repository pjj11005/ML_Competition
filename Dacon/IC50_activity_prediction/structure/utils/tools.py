import torch


# 평가 함수
def evaluate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids, attention_mask, features)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()

    model.train() # 다시 train 으로
    return val_loss / len(val_loader)
