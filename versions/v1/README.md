# Pharma-Augment: 요양급여 심사 도메인 질문 생성기

한국 요양급여 심사 도메인의 엑셀 데이터를 기반으로 학습용 질문을 생성하고 증강하는 도구입니다.

## 주요 기능

- 엑셀 파일에서 요양급여 심사 데이터 로드
- OpenAI/Claude API를 사용한 질문 생성
- 기본 5문형 + 증강 질문 자동 생성 (조항당 5~20개)
- 중복 제거 및 품질 필터링
- JSONL 형식 출력
- 비동기 처리로 높은 성능

## 설치

### Windows
```cmd
install_packages.bat
```

### Linux/Mac
```bash
chmod +x install_packages.sh
./install_packages.sh
```

### 수동 설치
```bash
pip install -r requirements.txt
```

## 환경 설정

`.env` 파일에 API 키를 설정하세요:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_claude_api_key_here  # Claude 사용 시
```

## 사용법

### 기본 사용

```bash
python generate_questions.py \
  --excel "data/요양심사약제_후처리_v2.xlsx" \
  --sheet "Sheet1" \
  --out "questions.jsonl" \
  --provider "openai" \
  --model "gpt-4o-mini" \
  --concurrency 6 \
  --max_aug 15 \
  --seed 20250902
```

### 옵션 설명

- `--excel`: 엑셀 파일 경로 (필수)
- `--sheet`: 시트명 (기본: Sheet1)
- `--out`: 출력 파일명 (기본: questions.jsonl)
- `--provider`: API 제공자 (openai/claude, 기본: openai)
- `--model`: 모델명 (기본: gpt-4o-mini)
- `--concurrency`: 동시 실행 수 (기본: 6)
- `--max_aug`: 최대 증강 질문 수 (기본: 15)
- `--seed`: 랜덤 시드 (기본: 20250902)
- `--base5_only`: 기본 5문형만 생성
- `--print-sample N`: N개 샘플 질문 출력

### 예시

```bash
# 샘플 출력과 함께 실행
python generate_questions.py \
  --excel "data/요양심사약제_후처리_v2.xlsx" \
  --print-sample 5

# Claude 사용
python generate_questions.py \
  --excel "data/요양심사약제_후처리_v2.xlsx" \
  --provider "claude" \
  --model "claude-3-sonnet-20240229"

# 기본 5문형만 생성
python generate_questions.py \
  --excel "data/요양심사약제_후처리_v2.xlsx" \
  --base5_only
```

## 입력 데이터 형식

엑셀 파일은 다음 컬럼을 포함해야 합니다:

- `약제분류번호`: 약제 분류 번호 (선택)
- `약제분류명`: 약제 분류명 (선택)
- `구분`: 조항 제목
- `세부인정기준 및 방법`: 본문 내용

## 출력 형식

JSONL 파일로 저장되며, 각 줄은 하나의 조항에 대한 JSON 객체입니다:

```json
{
  "clause_id": "간장용제_b54ee5da",
  "group_id": "간장용제",
  "title": "[일반원칙] 간장용제",
  "title_clean": "간장용제",
  "category": "일반원칙",
  "code": null,
  "code_name": null,
  "questions": [
    "간장용제의 요양급여 인정 범위는 무엇인가요?",
    "간장용제를 투여하기 위한 기준은 무엇인가요?",
    "..."
  ],
  "meta": {
    "source_sheet": "Sheet1",
    "mapping_confidence": 0.95,
    "dedup_rule": "rapidfuzz>=90",
    "neg_ratio": 0.0,
    "version": "qgen_v1.2",
    "seed": 20250902
  }
}
```

## 질문 생성 규칙

### 기본 5문형
1. **정의/범위**: "~의 적용 범위는 무엇인가요?"
2. **요건/기준**: "~의 인정 요건은 무엇인가요?"
3. **제외/불인정**: "~에서 제외되는 경우는 무엇인가요?"
4. **증빙/서류**: "~을 위해 필요한 서류는 무엇인가요?"
5. **엣지/경계**: "재시술/합병증 등의 경우 기준은 어떻게 적용되나요?"

### 증강 질문
- WH 다양화 (무엇/언제/어디서/누가/어떤 조건/어떻게/왜)
- 문체 변형 (~인가요/~해야 하나요/~가능한가요/~허용되나요/~인정되나요)
- 길이 변형 (짧은 요점형 ↔ 구체 조건형)
- 주체/시점 명시 (신청자/의료기관/담당부서, 최초/재시술/추적관찰)
- 조건 조합 (~인 경우에도 인정되나요? / ~이면 제외되나요?)

## 품질 관리

### 자동 필터링
- 금지어 제거 (추정, 일반적으로, 대체로, 관행상, 아마도 등)
- 외부 지식 의심 키워드 제거
- 길이 필터링 (15~180자)
- 중복 제거 (rapidfuzz ≥ 90% 유사도)

### 후처리
- 정규화 (공백/중복기호/말줄임 보정)
- 문장부호 통일 (물음표 추가)
- 5개 미만 시 재생성 시도

## 로그 및 감사

- `question_generation.log`: 실행 로그
- `audit_log.csv`: 성능 감사 로그 (토큰 사용량, 실행 시간 등)

## 성능 팁

- 동시성 6~8 권장 (API 제한 고려)
- 긴 조항(6000자 초과)은 자동 분할
- 실패 시 자동 재시도 (지수형 백오프)
- temperature=0.5, top_p=0.9로 다양성 확보

## 테스트

10행 샘플로 테스트:
```bash
python generate_questions.py \
  --excel "data/sample_요양심사약제_후처리_v2.xlsx" \
  --print-sample 3
```

## 문제 해결

### 일반적인 오류
1. **API 키 오류**: `.env` 파일의 API 키 확인
2. **엑셀 파일 오류**: 컬럼명 및 파일 경로 확인  
3. **메모리 부족**: concurrency 값 낮추기
4. **Rate Limit**: concurrency 값 낮추거나 재시도 대기

### 로그 확인
```bash
tail -f question_generation.log
```

## 라이선스

MIT License