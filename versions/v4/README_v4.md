# Pharma-Augment V4 실행 가이드

V4는 prompt_v4.md 스펙에 따른 **약제**와 **고시** 전용 질문 생성기입니다.

## 🎯 주요 특징

- **대명사 완전 차단**: "이/그/해당/본/동 + (약/약제/제제/제품)" 금지
- **엄격한 라벨링**: POSITIVE/NEGATIVE/HARD_NEGATIVE
- **품질 점수 시스템**: S_q ≥ 0.75만 채택
- **고정 출력 형식**: 제출용 엑셀 파일
- **실시간 진행상황**: tqdm 프로그레스 바

## 📂 입력 파일 준비

### 약제 데이터
파일: `C:\Jimin\Pharma-Augment\data\요양심사약제_후처리_v2.xlsx`
- 약제분류번호
- 약제 분류명  
- 구분
- 세부인정기준 및 방법

### 고시 데이터
파일: `C:\Jimin\Pharma-Augment\data\고시.xlsx`
- 고시번호
- 고시명칭
- 변경후 내용

## 🚀 실행 방법

### 1. 환경 설정
```bash
# V4 디렉토리로 이동
cd C:\Jimin\Pharma-Augment\versions\v4

# 필요한 패키지 설치
pip install -r requirements_v4.txt

# API 키 확인 (.env 파일)
# OPENAI_API_KEY=your_key_here
```

### 2. 약제 질문 생성
```bash
python drug_generator_v4.py --excel "C:\Jimin\Pharma-Augment\data\요양심사약제_후처리_v2.xlsx" --out drug_questions_v4.xlsx
```

**출력 파일**: `drug_questions_v4.xlsx`
- 컬럼: 약제분류번호, 약제 분류명, 구분, 세부인정기준 및 방법, question, 라벨

### 3. 고시 질문 생성
```bash
python notice_generator_v4.py --excel "C:\Jimin\Pharma-Augment\data\고시.xlsx" --out notice_questions_v4.xlsx
```

**출력 파일**: `notice_questions_v4.xlsx`
- 컬럼: 고시번호, 고시명칭, 변경후 내용, question, 라벨

## ⚙️ 고급 옵션

### 약제 생성기 옵션
```bash
python drug_generator_v4.py \
  --excel "데이터파일.xlsx" \
  --sheet "Sheet1" \
  --out "결과파일.xlsx" \
  --provider openai \
  --model gpt-4o-mini \
  --concurrency 6 \
  --seed 20250903
```

### 고시 생성기 옵션  
```bash
python notice_generator_v4.py \
  --excel "고시파일.xlsx" \
  --sheet "Sheet1" \
  --out "고시결과.xlsx" \
  --provider openai \
  --model gpt-4o-mini \
  --concurrency 6 \
  --seed 20250903
```

## 📊 실행 모니터링

실행 중 다음과 같은 정보를 확인할 수 있습니다:

1. **진행상황 바**: `질문 생성 중: 45%|████▌     | 123/273 [02:15<02:45, 약제/s]`
2. **실시간 로그**: 각 행별 질문 생성 완료 상태
3. **품질 검증**: 대명사 검출, 길이 위반, 점수 미달 등 필터링 로그
4. **최종 통계**: 생성된 질문 수, 라벨 분포

## 📋 출력 파일 구조

### 약제 결과 (drug_questions_v4.xlsx)
| 약제분류번호 | 약제 분류명 | 구분 | 세부인정기준 및 방법 | question | 라벨 |
|-------------|-------------|------|---------------------|----------|------|
| A01AD02 | Tacrolimus 제제 | ... | ... | Tacrolimus 제제의 조혈모세포이식 급여 범위는? | POSITIVE |

### 고시 결과 (notice_questions_v4.xlsx)
| 고시번호 | 고시명칭 | 변경후 내용 | question | 라벨 |
|----------|----------|-------------|----------|------|
| 보험 제2025-129호 | ... | ... | 해당 고시의 적용 대상군은? | POSITIVE |

## 🔍 품질 보장

- **S_q 점수**: 길이(15-70자), WH형, 단일논점, 원문중첩 종합 평가
- **라벨 검증**: POSITIVE ≥ 60%, HARD_NEGATIVE 10~25%, NEGATIVE 10~25%
- **대명사 차단**: 정규식 기반 100% 필터링
- **감사 로그**: `audit_log_drug_v4.csv`, `audit_log_notice_v4.csv`

## ❓ 문제 해결

### 일반적인 오류
1. **API 키 오류**: `.env` 파일에 `OPENAI_API_KEY` 확인
2. **파일 경로 오류**: 절대 경로 사용 권장
3. **메모리 부족**: `--concurrency` 값을 3-4로 낮춤
4. **Rate limit**: 잠시 대기 후 자동 재시도

### 로그 파일 위치
- `drug_generation_v4.log`
- `notice_generation_v4.log`  
- `audit_log_drug_v4.csv`
- `audit_log_notice_v4.csv`

## 🏃‍♂️ 빠른 실행 (기본 설정)

```bash
# 약제 + 고시 한번에 실행
cd C:\Jimin\Pharma-Augment\versions\v4

# 약제
python drug_generator_v4.py --excel "../../data/요양심사약제_후처리_v2.xlsx"

# 고시  
python notice_generator_v4.py --excel "../../data/고시.xlsx"
```

실행 완료 후 현재 디렉토리에 결과 파일들이 생성됩니다.