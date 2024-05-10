# 고객 대출등급 분류 AI 해커톤

## 대회 요약
- 데이터 출처 : [https://dacon.io/competitions/official/236214/data]

- 주최: 데이콘
- 주관: 데이콘
- 주제: 고객의 대출등급을 예측하는 AI 알고리즘 개발
- 평가 산식 : Macro F1
- 기간: 2024.01.15 ~ 2024.02.05
- 팀 구성: 개인 참여
- 상금: 대회 1등부터 3등까지는 수상 인증서(Certification)가 발급

## 진행 내용
- 수치형 데이터의 0값들 처리, feature engineering에 중점을 두고 전처리 작업 진행했다.

- 학습 모델은 가장 성능이 좋았던 `XGBClassifier`, `LGBMClassifier` 두 모델을 이용하여 feature selection, 하이퍼파라미터 튜닝을 진행했다.

- 최종 제출은 특성 중요도 최상위 5개 특성 + `LGBMClassifier` 하이퍼파라미터 튜닝 CV 10회 제출이었다. 