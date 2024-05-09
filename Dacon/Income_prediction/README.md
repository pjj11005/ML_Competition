# Dacon_Customer_Loan_Rating_Classification


## 대회 요약
- 데이터 출처 : [https://dacon.io/competitions/official/236230/data]

- 주최: 데이콘
- 주관: 데이콘
- 주제: 개인 특성 데이터를 활용하여 개인 소득 수준을 예측하는 AI 모델 개발
- 평가 산식 : RMSE
- 기간: 2024.03.11 ~ 2024.04.08
- 팀 구성: 개인 참여
- 상금: 대회 1등부터 3등까지는 수상 인증서(Certification)가 발급

## 진행 내용

>다른 일정으로 인해 제대로 대회에 참여하지 못했다...

- 학습 모델은 가장 성능이 좋았던 `LGBMClassifier` 모델을 이용하여 하이퍼파라미터 튜닝을 진행
- 학습 속도 및 성능 향상을 위해 feature importance 기반으로 상위 11개 변수만 사용
- 수치형 변수들의 0값들, 범주형 변수의 'unknown' 값들의 처리를 신경 썼다면 더 좋은 성과를 보였을 것 같다