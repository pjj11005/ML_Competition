# 제2회 AI 신약개발 경진대회

## 대회 요약
- 데이터 출처 : [https://dacon.io/competitions/official/236336/data]

- 주최/주관: 한국제약바이오협회 AI신약융합연구원
- 후원 : 보건복지부, 한국보건산업진흥원, 대웅제약
- 운영: 데이콘
- 상금: 총 2200만원
- 주제: IRAK4 IC50 활성 예측 모델 개발
- 평가 산식 : 0.5 X (1 - min(Normalized RMSE, 1)) + 0.5 X Correct Ratio
- 기간: 2024.08.05 ~ 2024.09.25
- 팀 구성: 개인 참여

## 진행 내용
- scikit-learn 머신러닝 모델들 publice score 0.53이 한계 → BERT기반 모델들로 변경
- SMILES는 **DeepChem/ChemBERTa-77M-MTR** 모델로 분자 및 원자 특성은 추가 신경망으로 학습 후 결합
- **50회 학습 후 예측 결과 생성 →** public : **0.56492 /** private : **0.61647**
- 조합한 분자, 원자 특성들을 가감하며 실험하거나 모델의 layer, optimizer 등의 파라미터 조절 필요