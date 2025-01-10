import torch
import torch.nn as nn


# 모델 설계: ChemBERT + 수치형 처리 layers
class ChemBERTpIC50Predictor(nn.Module):
    def __init__(self, bert_model, num_features):
        super(ChemBERTpIC50Predictor, self).__init__()
        self.bert_model = bert_model

        # 적은 양의 데이터셋으로 작은 layer + 규제 적용
        self.fc_features = nn.Sequential(
            nn.Linear(num_features, 128), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(128, 64))

        # 최종 layer = ChemBERT output + features
        self.fc_combined = nn.Sequential(
            nn.Linear(self.bert_model.config.hidden_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask, features):
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)[1]  # Pooled output
        features_output = self.fc_features(features)
        combined = torch.cat([bert_output, features_output], dim=1)
        output = self.fc_combined(combined)
        return output
