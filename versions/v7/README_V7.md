# V7 질문 생성기 - ReasonIR 기반 멀티밴드

## 🎯 핵심 개선사항

V6 → V7 주요 변화 (ReasonIR 논문 반영):

### 1. 길이 밴드 시스템 (키워드 과적합 방지)
- **SR (25-80자) 60%**: 단답형, 키워드 중심
- **MR (80-160자) 25%**: 중간 복합 질문  
- **LR (200-600자) 15%**: 시나리오 기반 맥락 질문

### 2. 멀티턴 생성으로 맥락 이해 강화
- **LR 밴드**: `시나리오(2-4문장) → 질문` 패턴
- 단순 키워드 매칭 탈피 → 임상 맥락 이해
- 300-2000단어 VL 데이터 범위 (ReasonIR 방식)

### 3. 앵커팩 수집 시스템
- 동일 `anchor_id`로 SR/MR/LR을 묶어서 수집
- 멀티뷰 학습 데이터 지원
- JSONL 형태로 앵커팩 별도 저장

### 4. 밴드별 차등 대명사 규칙
- **SR/MR**: 완전 금지 (기존 유지)
- **LR**: 시나리오 서술부만 조건부 허용, 질문부는 금지

### 5. 밴드 내 라벨 비율 유지
- 전체: **POS:HN:EN = 6:3:0** 
- 각 밴드 내에서도 동일 비율 적용 (±1 허용)

## 🏗️ 아키텍처

### 길이 밴드별 생성 전략

#### SR 밴드 (Short Range)
```
프롬프트: 25-80자 단답형 질문
예시: "트라스투주맙의 급여기준은?"
특징: 키워드 중심, 직접적
```

#### MR 밴드 (Medium Range)  
```
프롬프트: 80-160자 복합 질문
예시: "HER2 양성 유방암에서 허셉틴 6mg/kg 투여시 모니터링 항목은?"
특징: 조건부 복합, 구체적
```

#### LR 밴드 (Long Range)
```
프롬프트: 200-600자 시나리오→질문
예시: "60세 여성이 HER2 양성 유방암으로 진단받았다. 
      기존 항암치료 후 재발이 확인되었고, 심기능은 정상이다. 
      이 경우 트라스투주맙 급여 적용이 가능한가?"
특징: 맥락적, 임상상황 기반
```

### Hard Negative 생성
- **앵커 기반**: 상위 POS 질문을 앵커로 선택
- **단일 Facet 변형**: 1개 차원만 의도적으로 변경
- **리라이트**: LLM으로 자연스러운 질문으로 변환

### 후처리 파이프라인
1. 길이 밴드별 필터링
2. 밴드별 차등 대명사 검증
3. 구체성 확인 (숫자/단위/정책어)
4. 문두 다양성 체크 (30% 룰)
5. RapidFuzz 중복 제거 (82% 임계값)
6. 밴드별 라벨 비율 정규화

## 🚀 사용법

### 전체 실행
```bash
cd versions/v7
python drug_generator_v7.py
```

### 테스트 실행 (Mock)
```bash
cd versions/v7  
python test_v7.py
```

### 배치 실행
```bash
# 배치 파일 생성 필요
versions/v7/run_v7.bat
```

## 📊 출력 형식

### 기본 엑셀 출력
```
약제분류번호 | 약제 분류명 | 구분 | 세부인정기준 및 방법 | question | 라벨
```

### 앵커팩 JSONL 출력
```json
{"anchor_id": "uuid", "band": "SR", "question": "질문", "doc_slice_id": "A001_0", "label": "POSITIVE"}
{"anchor_id": "uuid", "band": "MR", "question": "질문", "doc_slice_id": "A001_0", "label": "POSITIVE"}  
{"anchor_id": "uuid", "band": "LR", "question": "질문", "doc_slice_id": "A001_0", "label": "POSITIVE"}
```

## ⚙️ 설정 가능한 옵션

### 밴드 비율 조정
```python
config.bands[LengthBand.SR].ratio = 0.5  # 50%
config.bands[LengthBand.MR].ratio = 0.3  # 30% 
config.bands[LengthBand.LR].ratio = 0.2  # 20%
```

### 라벨 비율 조정
```python
config.pos_ratio = 6.0
config.hn_ratio = 3.0  
config.en_ratio = 1.0  # EN 활성화 (실험시)
```

### 길이 범위 조정
```python
config.bands[LengthBand.SR].min_chars = 20
config.bands[LengthBand.SR].max_chars = 100
```

## 🔧 V6 대비 개선점

| 항목 | V6 | V7 | 개선효과 |
|------|----|----|---------|
| 길이 다양성 | 단일 범위 | 3단계 밴드 | 키워드 과적합 방지 |
| 맥락 이해 | 키워드 기반 | 시나리오 기반 | 임상맥락 강화 |
| 앵커팩 | 미지원 | 지원 | 멀티뷰 학습 |
| 대명사 규칙 | 일괄 금지 | 밴드별 차등 | 자연스러움↑ |
| 라벨 균형 | 전체만 | 밴드별도 | 세밀한 제어 |

## 🧪 실험 모드

### A/B 테스트 지원
```python
# 기본: POS:HN:EN = 6:3:0
# 실험: POS:HN:EN = 6:3:1
config.en_ratio = 1.0
```

### 밴드 비율 실험
```python  
# Conservative: SR 많이
sr_heavy = {"SR": 0.7, "MR": 0.2, "LR": 0.1}

# Aggressive: LR 많이 (맥락 중심)
lr_heavy = {"SR": 0.4, "MR": 0.3, "LR": 0.3}
```

## 📁 파일 구조

```
versions/v7/
├── drug_generator_v7.py      # 메인 생성기
├── test_v7.py               # 테스트 스크립트
├── README_V7.md             # 이 문서
├── run_v7.bat               # 실행 배치파일
└── [결과 파일들]
    ├── drug_questions_v7.xlsx
    ├── drug_questions_v7_anchorpack.jsonl
    └── drug_generation_v7.log
```

## 🎯 기대 효과

1. **키워드 과적합 방지**: 다양한 길이로 표현 다양성 확보
2. **맥락적 이해 향상**: LR 시나리오로 임상 상황 반영
3. **학습 데이터 품질**: 앵커팩으로 멀티뷰 학습 지원
4. **생성 안정성**: 밴드별 후처리로 일관된 품질
5. **확장성**: 밴드/라벨 비율 실험적 조정 가능

ReasonIR 논문의 핵심 아이디어를 의료보험 도메인에 최적화하여 적용한 V7입니다! 🚀