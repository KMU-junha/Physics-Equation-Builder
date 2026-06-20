# Meta-Symbolic Evolution Trace (Log)

### 데이터 진단
- 감지된 데이터 형태: `SYMBOLIC`

---
## 1. 공리 연역 (Axiomatic Deduction) 판정 결과
`Integral(Unknown_Term*f_t, (t, 0, oo))`
`Integral(Unknown_Term*f_t, (t, -oo, oo))`

---
## 2. 동적 유전자 풀 (Dynamic Gene Pool)
- 변수 및 상수: `[0, 1, 2, 's', 'b', 'a', 't', -1]`

---
## 3. 탐색 하이퍼파라미터 (Search Hyperparameters)
- 필수 도메인 변수: `['t', 's']` (2개)
- 최소 강제 탐색 깊이: `1`
- 세대별 개체 수(Population): `1000`
- 층위별 최대 세대 수: `3`회

---
## [Depth 1] 점진적 심층 탐색 시작

- [세대 0 완료] 누적 탐색: `1000`개 | 세대 최고점: `10.7591` | 중복 캐시 히트율: `8.3`%
### [전역 최고해 갱신 (Global Best Update)]
- 진행 스텝: Depth 1 | Gen 0
- 적합도 점수(Fitness Score): 10.7591
`Integral(f_t*log(sin(s*t) - 1), (t, -oo, oo))`

- [세대 1 완료] 누적 탐색: `2000`개 | 세대 최고점: `100.0000` | 중복 캐시 히트율: `26.8`%
### [전역 최고해 갱신 (Global Best Update)]
- 진행 스텝: Depth 1 | Gen 1
- 적합도 점수(Fitness Score): 100.0000
`Integral(f_t*exp(-s*t), (t, 0, oo))`


---
## 4. 최종 탐색 요약 (Final Output)
- 총 탐색 소요 시간: `11.47`초
- 수식 복잡도 (AST Node Size): `6`

### [추론된 물리 방정식 (Physical Operator)] Score: 100.0000
`Integral(f_t*exp(-s*t), (t, 0, oo))`


---
## 5. 결과 해석 및 탐색 경로 분석

### 5.1 탐색 경로 (Discovery Path)
- 방법: 점진적 심층 탐색 + 빔 서치 GA (Successive Deepening + Beam Search)
- 원리: 최소 깊이부터 시작하여 빔 서치로 우수 커널을 씨앗으로 점진 확장,
  상위 엘리트에 대해 기호 증명(Symbolic Proof)으로 완전 일치 여부 검증
- 소요 시간: `11.466초`
- 수식 평가 수: `2000개`
- 기호 증명: ✅ 성공 — 닫힌 형태(Closed Form) 완전 일치 확인

### 5.2 연산자 구조 분석 (Structural Analysis)
- 추론된 연산자: `Integral(f_t*exp(-s*t), (t, 0, oo))`
- 구조: 적분형 연산자 (Integral Operator)
  - 적분 변수: `t`
  - 적분 구간: `[0, oo)`
  - 피적분함수: `f_t * exp(-s*t)`
  - 커널 (Kernel): `exp(-s*t)`

