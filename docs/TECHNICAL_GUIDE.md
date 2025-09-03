# Pharma-Augment 기술 가이드

## 📋 개요

Pharma-Augment 프로젝트의 기술적 구현 세부사항, 아키텍처, 그리고 개발 과정에서 발견한 핵심 이슈와 해결책을 정리한 기술 문서입니다.

## 🏗️ 프로젝트 아키텍처

### 디렉토리 구조
```
Pharma-Augment/
├── versions/              # 버전별 구현체
│   ├── v1/               # 기본 질문 생성
│   ├── v2/               # 임베딩 최적화  
│   ├── v3/               # 대명사 차단 + 이름비율
│   ├── v4/               # 100% 커버리지 보장
│   └── v5/               # 품질 + 커버리지 균형
├── data/
│   ├── inputs/           # 원본 엑셀 파일
│   └── outputs/          # 생성 결과 파일
├── utils/                # 유틸리티 스크립트  
└── docs/                 # 문서 및 프롬프트
```

### 핵심 컴포넌트

1. **데이터 로더** (`load_excel_data`)
   - 엑셀 파일 읽기 및 컬럼 매핑
   - UTF-8 인코딩 문제 해결
   - 필수 필드 검증

2. **전처리기** (`preprocess_data`)
   - NaN 값 처리
   - 텍스트 정규화
   - clause_id 생성

3. **프롬프트 엔진** (`create_prompt`)
   - 버전별 맞춤 프롬프트 생성
   - 동적 파라미터 삽입
   - 품질 기준 정의

4. **API 클라이언트** (`call_api`)
   - 비동기 HTTP 요청
   - Rate limit 처리
   - 재시도 로직

5. **후처리기** (`post_process`)
   - 정규식 기반 검증
   - 품질 점수 계산
   - 필터링 및 정제

## 🔧 핵심 기술 구현

### 1. 대명사 차단 시스템

**정규식 패턴**:
```python
PRONOUN_RE = re.compile(r"(이|그|해당|본|동)\s?(약|약제|제제|제품)|이것|그것")
```

**검증 로직**:
```python
def validate_pronoun(text: str) -> bool:
    """대명사 검증"""
    return not self.PRONOUN_RE.search(text)

# 사용 예시
for question in questions:
    if not validate_pronoun(question['text']):
        logger.warning(f"대명사 검출: {question['text']}")
        continue  # 해당 질문 제거
```

**검출 예시**:
```
❌ "이 약제의 사용법은?" → 대명사 검출
❌ "해당 제제의 기준은?" → 대명사 검출  
✅ "Tacrolimus의 사용법은?" → 통과
```

### 2. 이름 사용 비율 시스템

**추출 로직**:
```python
def extract_name_slots(gubun: str) -> Tuple[str, List[str]]:
    """구분 필드에서 약제명 추출"""
    # [일반원칙] 제거
    clean_gubun = re.sub(r'\[.*?\]\s*', '', gubun)
    
    # 주 약제명 추출 (괄호 앞)
    main_match = re.search(r'^([^(]+)', clean_gubun)
    main_name = main_match.group(1).strip() if main_match else ""
    
    # 품명 추출 (품명: 뒤)
    brand_match = re.search(r'품\s*명\s*[:：]\s*([^)]+)', clean_gubun)
    brand_names = []
    if brand_match:
        brand_text = brand_match.group(1).strip()
        brand_names = re.split(r'[·/]+', brand_text)
        brand_names = [name.strip() for name in brand_names if name.strip()]
    
    return main_name, brand_names
```

**비율 계산**:
```python
def calculate_ratios(brand_names: List[str]) -> Dict[str, Tuple[float, float]]:
    """브랜드명 개수별 비율 계산"""
    num_brands = len(brand_names)
    
    if num_brands == 0:
        return {"MAIN": (0.70, 0.80), "BRAND": (0.0, 0.0), "BOTH": (0.20, 0.30)}
    elif num_brands == 1:  
        return {"MAIN": (0.35, 0.45), "BRAND": (0.30, 0.40), "BOTH": (0.20, 0.30)}
    else:
        return {"MAIN": (0.30, 0.40), "BRAND": (0.30, 0.40), "BOTH": (0.20, 0.30)}
```

### 3. 품질 점수 시스템 (V4)

**S_q 점수 계산**:
```python
def calculate_question_score(text: str, content: str) -> float:
    """문항 점수 S_q 계산 (0~1)"""
    
    # 1. 길이 점수 (45자 최적)
    length = len(text)
    length_score = max(0, min(1, 1 - abs(length - 45) / 30))
    
    # 2. WH 점수
    wh_patterns = r'(무엇|어떤|언제|어떻게|왜|누가|어디|몇)'
    if re.search(f'^{wh_patterns}', text):
        wh_score = 1.0
    elif re.search(wh_patterns, text):
        wh_score = 0.6
    else:
        wh_score = 0.2
    
    # 3. 단일 논점 점수
    multi_issue_count = text.count(',') + text.count('및') + text.count('/')
    single_issue = 1.0 if multi_issue_count <= 1 else 0.3
    
    # 4. 대명사 패널티
    pronoun_penalty = -1.0 if self.PRONOUN_RE.search(text) else 0.0
    
    # 5. 원문 중첩 점수
    text_tokens = set(re.findall(r'\w+', text))
    content_tokens = set(re.findall(r'\w+', content))
    overlap = len(text_tokens & content_tokens) / len(text_tokens) if text_tokens else 0.0
    
    # 최종 점수
    s_q = (0.25 * length_score + 0.25 * wh_score + 
           0.25 * single_issue + 0.25 * overlap + pronoun_penalty)
    
    return max(0.0, min(1.0, s_q))
```

### 4. Fallback 시스템 (V4)

**3단계 재시도 로직**:
```python
async def generate_questions_for_drug(self, session, row_data, row_idx):
    """3단계 fallback으로 100% 성공 보장"""
    
    validated_questions = []
    
    # 1차: 엄격한 기준
    try:
        prompt = self.create_strict_prompt(row_data)
        questions = await self.call_api(session, prompt)
        validated_questions = self.strict_post_process(questions)
        if len(validated_questions) >= 5:
            return validated_questions
    except Exception as e:
        logger.warning(f"1차 시도 실패: {e}")
    
    # 2차: 완화된 기준
    try:
        prompt = self.create_relaxed_prompt(row_data)
        questions = await self.call_api(session, prompt)
        validated_questions = self.relaxed_post_process(questions)
        if len(validated_questions) >= 5:
            return validated_questions  
    except Exception as e:
        logger.warning(f"2차 시도 실패: {e}")
    
    # 3차: 기본 질문 생성 (100% 보장)
    return self.create_basic_questions(row_data)
```

### 5. V5 구체성 검증 시스템

**구체성 패턴 매칭**:
```python
def validate_concreteness(text: str) -> bool:
    """V5 구체성 검증"""
    concrete_patterns = [
        r'\d+',                           # 숫자 포함
        r'(개월|일|회|mg|mL|U/L)',        # 의료 단위
        r'(환자|처방|투여|사용|적용)',      # 의료 용어
        r'(기준|조건|절차|방법|사항)',      # 구체적 명사
    ]
    
    return any(re.search(pattern, text) for pattern in concrete_patterns)

# V5에서 사용
if not validate_concreteness(question_text):
    logger.warning(f"V5 구체성 부족: {question_text}")
    continue
```

## 🔍 주요 이슈와 해결책

### 1. 인코딩 문제

**문제**: Windows 환경에서 UTF-8 한글 처리 오류
```
UnicodeEncodeError: 'cp949' codec can't encode character
```

**해결책**:
```python
# 로깅 핸들러에서 UTF-8 명시
logging.FileHandler('log.log', encoding='utf-8')

# 파일 저장 시 인코딩 지정
df.to_csv(path, encoding='utf-8-sig')  # BOM 포함
df.to_excel(path, engine='openpyxl')   # 엑셀은 openpyxl 사용
```

### 2. [괄호] 패턴 오염

**문제**: 원본 데이터의 [일반원칙] 등이 질문에 포함됨

**해결 과정**:
1. **문제 발견**: 생성된 질문에 "[일반원칙] 간장제제의..." 형태
2. **원인 분석**: 원본 데이터의 구분 필드에 괄호 패턴 존재
3. **해결책 구현**: 
   - `data_cleaner.py`: 원본 데이터 정리
   - `question_only_cleaner.py`: 질문 필드만 정리

**정리 스크립트**:
```python
def clean_bracket_patterns(text):
    """괄호 패턴 제거"""
    text = re.sub(r'\[.*?\]\s*', '', text)  # [패턴] 제거
    return text.strip()

# 원본 데이터 정리
df['구분'] = df['구분'].apply(clean_bracket_patterns)

# 생성된 질문만 정리 (구분 필드는 유지)
df['question'] = df['question'].apply(clean_bracket_patterns)
```

### 3. API Rate Limit 처리

**문제**: OpenAI API 429 에러로 인한 생성 중단

**해결책**:
```python
@backoff.on_exception(
    backoff.expo, 
    (aiohttp.ClientError, asyncio.TimeoutError), 
    max_tries=3
)
async def call_api(self, session, prompt):
    """백오프 재시도 로직"""
    
    async with session.post(url, json=payload) as response:
        if response.status == 429:
            logger.warning("Rate limit, retrying...")
            await asyncio.sleep(3)  # 잠시 대기
            raise aiohttp.ClientError("Rate limit")
        
        response.raise_for_status()
        return await response.json()
```

### 4. 동시성 제어

**문제**: 너무 많은 동시 요청으로 인한 시스템 부하

**해결책**:
```python
# 세마포어로 동시성 제어
self.semaphore = asyncio.Semaphore(concurrency)

async def generate_questions(self, session, data):
    async with self.semaphore:  # 최대 concurrency개만 동시 실행
        # API 호출 로직
        pass
```

### 5. 진행상황 시각화

**문제**: 대용량 데이터 처리 시 진행상황 파악 어려움

**해결책**:
```python
from tqdm.asyncio import tqdm

# 비동기 진행바
self.progress_bar = tqdm(total=len(data), desc="질문 생성 중", unit="약제")

async def process_item(self, item):
    # 작업 처리
    result = await self.generate_questions(item)
    
    # 진행상황 업데이트
    if self.progress_bar:
        self.progress_bar.update(1)
    
    return result
```

## 📊 성능 최적화

### 1. 배치 처리 최적화

```python
# 비효율적: 순차 처리
for item in data:
    result = await process_item(item)

# 효율적: 비동기 배치 처리  
tasks = [process_item(item) for item in data]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

### 2. 메모리 사용량 최적화

```python
# 대용량 데이터 스트림 처리
def process_large_file(file_path):
    chunk_size = 1000
    
    for chunk in pd.read_excel(file_path, chunksize=chunk_size):
        yield chunk  # 제너레이터로 메모리 절약
```

### 3. 캐싱 시스템

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def extract_drug_info(gubun: str):
    """자주 사용되는 약제 정보 캐싱"""
    # 파싱 로직
    return main_name, brand_names
```

## 🧪 테스트 전략

### 1. 단위 테스트
```python
def test_pronoun_detection():
    """대명사 검출 테스트"""
    assert not validate_pronoun("이 약제의 사용법")  # 검출되어야 함
    assert validate_pronoun("Tacrolimus의 사용법")   # 통과되어야 함

def test_name_extraction():
    """약제명 추출 테스트"""  
    gubun = "Tacrolimus 제제 (품명: 프로그랍캅셀·주사)"
    main_name, brand_names = extract_name_slots(gubun)
    assert main_name == "Tacrolimus 제제"
    assert brand_names == ["프로그랍캅셀", "주사"]
```

### 2. 통합 테스트
```python
def test_end_to_end_generation():
    """전체 프로세스 테스트"""
    sample_data = load_sample_data()
    generator = DrugGeneratorV5()
    
    results = asyncio.run(generator.generate_all_questions(sample_data))
    
    assert len(results) > 0
    assert all(validate_pronoun(r['question']) for r in results)
    assert all(25 <= len(r['question']) <= 80 for r in results)
```

## 📈 모니터링과 로깅

### 1. 감사 로그 시스템
```python
self.audit_log.append({
    'row_id': row_id,
    'questions_generated': len(questions),
    'avg_length': avg_length,
    'elapsed_ms': elapsed_time,
    'success': is_success,
    'provider': self.provider,
    'model': self.model,
    'version': 'v5'
})

# CSV로 저장하여 분석 용이
df_audit = pd.DataFrame(self.audit_log)
df_audit.to_csv('audit_log.csv')
```

### 2. 품질 메트릭 수집
```python
def calculate_quality_metrics(questions):
    """품질 메트릭 계산"""
    return {
        'avg_length': np.mean([len(q) for q in questions]),
        'pronoun_rate': sum(1 for q in questions if PRONOUN_RE.search(q)) / len(questions),
        'concrete_rate': sum(1 for q in questions if validate_concreteness(q)) / len(questions),
        'wh_rate': sum(1 for q in questions if re.search(r'^(무엇|어떤)', q)) / len(questions)
    }
```

## 🚀 배포 및 운영

### 1. 환경 설정
```bash
# requirements.txt
pandas>=2.0.0
aiohttp>=3.8.0  
backoff>=2.2.0
rapidfuzz>=3.0.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
tqdm>=4.65.0
openpyxl>=3.1.0

# .env 파일
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_claude_key_here
```

### 2. 도커화
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "drug_generator_v5.py", "--excel", "data.xlsx"]
```

### 3. 배치 스크립트
```bash
#!/bin/bash
# run_v5.sh

echo "V5 질문 생성 시작..."
python versions/v5/drug_generator_v5.py \
  --excel "data/cleaned_drug_data.xlsx" \
  --out "results/drug_questions_v5.xlsx" \
  --concurrency 4

echo "완료: results/drug_questions_v5.xlsx"
```

---

*최종 업데이트: 2025-09-03*  
*작성자: Claude Code Assistant*