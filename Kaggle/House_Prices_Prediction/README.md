# House Prices - Advanced Regression Techniques


## 대회 요약
- 데이터 출처 : [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data]

- 주최: Kaggle
- 종류: 연습
- 주제: 개인 특성 데이터를 활용하여 개인 소득 수준을 예측하는 AI 모델 개발
- 평가 산식 : RMSE

## 진행 내용

- 이상치 제거, 특성 조합 등의 전처리 과정 파이프라인 구축
- RandomSearchCV로 가장 성능 좋은 4개의 모델 선정 후, 4개 모델의 평균 모델 성능 평가
- 위의 4개의 모델(SVR, XGB, LGBM, CatBoost) Stacking 모델 생성
- 최종적으로 Stacking + XGB + LGBM(0.7 + 0.3 + 0.3)비율의 예측 결과 제출