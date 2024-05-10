# 저해상도 조류 이미지 분류 AI 경진대회

## 대회 요약
- 데이터 출처 : [https://dacon.io/competitions/official/236251/data]

- 주최/주관: 데이콘
- 주제: 저해상도 조류 이미지 분류 AI 알고리즘 개발
- 평가 산식 : Macro F1 Score
- 기간: 2024.04.08 ~ 2024.05.06
- 팀 구성: 팀 참여(4인)
- 상금: 대회 1등부터 5등까지는 수상 인증서(Certification)가 발급

## 진행 내용

- keras의 사전학습 모델들 public score 0.90이 한계 → Transformer 모델로 변경
- ViT large 16 모델  `epochs: 10, 초기 lr: 1e-5` → public : 0.9599
- microsoft/beit-large-patch16-224-pt22k-ft22k
    - `batch_size:64, gradient_accumulation_steps: 4, epochs: 10, learning_rate=5e-5` → public : 0.9705
- eva_large_patch14_196.in22k_ft_in22k_in1k
    - `batch_size:32, epochs: 8, learning_rate=1e-5` → **public : 0.9733**
- Stratified Kfold, 데이터 증강 등을 사용하여 성능을 향상을 시도해 봐야했다