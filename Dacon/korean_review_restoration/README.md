# 난독화된 한글 리뷰 복원 AI 경진대회

## 대회 요약
- 데이터 출처 : [https://dacon.io/competitions/official/236446/data]

- 주최/주관: 데이콘
- 주제: 난독화된 한글 리뷰를 복원하는 AI 알고리즘 개발
- 평가 산식 : 문자 단위의 F1 Score
- 기간: 2025.01.06 ~ 2025.02.28
- 팀 구성: 개인 참여
- 상금: 대회 1등부터 3등까지는 수상 인증서(Certification)가 발급

## 진행 내용

- 난독화된 한국어 리뷰 복원을 위해 `rtzr/ko-gemma-2-9b-it 9B` 모델을 `4비트 양자화`와 `LoRA(r=16, alpha=32)`로 최적화하여 사용
- 학습 효율성을 위해 `batch_size 2, gradient_accumulation_steps 16, lr 2e-4` 설정으로 `3 epochs` 학습 진행
- 추론 시 `temperature 0.2, top_p 0.9` 파라미터로 자연스러운 한국어 복원 결과 생성
- 완성된 모델을 `ko-gemma-2-9b-it-restoration` 이름으로 허깅페이스 허브에 공개하여 재사용성 확보