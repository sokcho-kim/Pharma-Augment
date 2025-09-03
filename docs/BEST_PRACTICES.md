# Pharma-Augment 베스트 프랙티스

## 📋 개요

Pharma-Augment 프로젝트를 효과적으로 사용하고 유지보수하기 위한 베스트 프랙티스와 권장사항을 정리한 문서입니다.

## 🎯 버전 선택 가이드

### 목적별 권장 버전

| 사용 목적 | 권장 버전 | 이유 |
|----------|-----------|------|
| **임베딩 모델 학습** | **V5** | 구체적이고 다양한 고품질 질문 |
| **100% 데이터 커버리지** | V4 | 모든 행에서 반드시 질문 생성 |
| **빠른 프로토타이핑** | V1 | 단순하고 빠른 기본 생성 |
| **레거시 호환성** | V2 | 기존 임베딩 모델과 호환 |

### 품질 vs 커버리지 트레이드오프

```
높은 품질 원할 때: V5 → V2 → V3 → V4 → V1
높은 커버리지 원할 때: V4 → V5 → V1 → V2 → V3
```

## 🔧 실행 전 체크리스트

### 1. 환경 준비
- [ ] Python 3.8+ 설치 확인
- [ ] 필요한 패키지 설치: `pip install -r requirements.txt`
- [ ] API 키 설정: `.env` 파일에 `OPENAI_API_KEY` 추가
- [ ] 입력 데이터 준비: 엑셀 파일 경로 확인

### 2. 데이터 검증
- [ ] 필수 컬럼 존재 확인: `구분`, `세부인정기준 및 방법`
- [ ] 한글 인코딩 문제 없는지 확인
- [ ] 괄호 패턴 정리 여부 확인: `[일반원칙]` 등

### 3. 리소스 계획
- [ ] API 사용량 한도 확인
- [ ] 동시성 수준 설정: `--concurrency` (권장: 4-6)
- [ ] 디스크 용량 확인: 결과 파일 저장 공간

## 💡 최적 실행 방법

### V5 실행 (권장)

```bash
# 기본 실행
python versions/v5/drug_generator_v5.py \
  --excel "data/cleaned_data.xlsx" \
  --out "results/v5_questions.xlsx"

# 최적화된 실행 (추천 설정)
python versions/v5/drug_generator_v5.py \
  --excel "data/cleaned_data.xlsx" \
  --out "results/v5_questions.xlsx" \
  --provider openai \
  --model gpt-4o-mini \
  --concurrency 4 \
  --seed 20250903
```

### 대용량 데이터 처리

```bash
# 단계별 처리 (1000행씩)
python utils/batch_processor.py \
  --input "large_data.xlsx" \
  --batch_size 1000 \
  --version v5

# 병렬 처리 (여러 프로세스)
python utils/parallel_runner.py \
  --input "large_data.xlsx" \
  --workers 4 \
  --version v5
```

## 📊 품질 관리 가이드

### 1. 생성 전 데이터 품질 체크

```python
# 데이터 품질 검증 스크립트 예시
def validate_input_data(df):
    """입력 데이터 품질 검증"""
    issues = []
    
    # 1. 필수 컬럼 확인
    required_cols = ['구분', '세부인정기준 및 방법']
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"필수 컬럼 누락: {col}")
    
    # 2. 빈 값 비율 확인
    for col in required_cols:
        if col in df.columns:
            empty_rate = df[col].isna().sum() / len(df)
            if empty_rate > 0.1:  # 10% 이상
                issues.append(f"{col} 빈 값 비율 높음: {empty_rate:.1%}")
    
    # 3. 괄호 패턴 확인
    if '구분' in df.columns:
        bracket_count = df['구분'].str.contains(r'\[.*?\]', na=False).sum()
        if bracket_count > 0:
            issues.append(f"괄호 패턴 발견: {bracket_count}개 - 사전 정리 필요")
    
    return issues

# 사용법
issues = validate_input_data(df)
if issues:
    print("⚠️  데이터 품질 이슈:")
    for issue in issues:
        print(f"  - {issue}")
    print("✅ 해결 후 다시 실행하세요.")
```

### 2. 생성 후 품질 평가

```python
def evaluate_generated_questions(questions):
    """생성된 질문 품질 평가"""
    metrics = {}
    
    # 기본 통계
    metrics['total_count'] = len(questions)
    metrics['avg_length'] = np.mean([len(q['question']) for q in questions])
    
    # 대명사 사용률 (0%가 목표)
    pronoun_pattern = re.compile(r"(이|그|해당|본|동)\s?(약|약제|제제|제품)|이것|그것")
    pronoun_count = sum(1 for q in questions if pronoun_pattern.search(q['question']))
    metrics['pronoun_rate'] = pronoun_count / len(questions)
    
    # 구체성 비율
    concrete_patterns = [r'\d+', r'(개월|일|회|mg)', r'(환자|처방|투여)']
    concrete_count = sum(1 for q in questions 
                        if any(re.search(p, q['question']) for p in concrete_patterns))
    metrics['concrete_rate'] = concrete_count / len(questions)
    
    # WH 질문 비율
    wh_count = sum(1 for q in questions 
                  if re.search(r'(무엇|어떤|언제|어떻게|왜)', q['question']))
    metrics['wh_rate'] = wh_count / len(questions)
    
    return metrics

# 품질 기준
QUALITY_THRESHOLDS = {
    'avg_length': (30, 60),     # 30-60자
    'pronoun_rate': (0, 0.01),  # 1% 미만
    'concrete_rate': (0.7, 1),  # 70% 이상
    'wh_rate': (0.8, 1)         # 80% 이상
}

def check_quality_standards(metrics):
    """품질 기준 달성 여부 확인"""
    results = {}
    
    for metric, (min_val, max_val) in QUALITY_THRESHOLDS.items():
        value = metrics[metric]
        passed = min_val <= value <= max_val
        results[metric] = {
            'value': value,
            'target': f"{min_val}-{max_val}",
            'passed': passed
        }
    
    return results
```

### 3. 품질 개선 가이드

**품질이 낮을 때 개선 방법**:

1. **평균 길이가 짧을 때** (< 30자)
   ```python
   # 프롬프트에 길이 제약 강화
   "반드시 30자 이상의 구체적이고 상세한 질문을 생성하라"
   
   # 온도 조정
   payload["temperature"] = 0.8  # 더 창의적으로
   ```

2. **구체성이 부족할 때** (< 70%)
   ```python
   # 프롬프트에 구체성 요구사항 추가
   "반드시 다음 중 하나 이상 포함: 구체적 수치, 의료용어, 환자군, 절차"
   ```

3. **대명사 사용률이 높을 때** (> 1%)
   ```python
   # 후처리에서 더 엄격하게 필터링
   if pronoun_pattern.search(question_text):
       logger.warning(f"대명사 검출: {question_text}")
       continue  # 질문 제거
   ```

## 🚨 문제 해결 가이드

### 자주 발생하는 오류와 해결책

#### 1. API 관련 오류

**에러**: `OpenAI API key not found`
```bash
# 해결책
echo "OPENAI_API_KEY=your_actual_key_here" >> .env
```

**에러**: `Rate limit exceeded`
```bash
# 해결책: 동시성 낮추기
python generate.py --concurrency 2  # 기본 6 → 2로 낮춤
```

#### 2. 데이터 관련 오류

**에러**: `KeyError: '구분'`
```python
# 해결책: 컬럼명 확인 및 매핑
print("실제 컬럼명:", df.columns.tolist())
# 필요시 수동 매핑
df = df.rename(columns={'실제컬럼명': '구분'})
```

**에러**: `UnicodeDecodeError`
```python
# 해결책: 인코딩 명시
df = pd.read_excel(file_path, encoding='utf-8-sig')
```

#### 3. 메모리 관련 오류

**에러**: `MemoryError`
```python
# 해결책: 배치 처리
def process_in_batches(df, batch_size=100):
    for i in range(0, len(df), batch_size):
        batch = df[i:i+batch_size]
        yield batch
        
# 사용법
for batch in process_in_batches(large_df):
    results = generator.process(batch)
```

### 성능 최적화 팁

#### 1. API 호출 최적화

```python
# 좋은 예: 비동기 배치 처리
tasks = [process_item(item) for item in data[:100]]  # 100개씩
results = await asyncio.gather(*tasks)

# 나쁜 예: 순차 처리
for item in data:
    result = await process_item(item)  # 너무 느림
```

#### 2. 메모리 사용 최적화

```python
# 좋은 예: 제너레이터 사용
def load_data_lazy(file_path):
    for chunk in pd.read_excel(file_path, chunksize=1000):
        yield chunk

# 나쁜 예: 전체 로드
df = pd.read_excel(huge_file)  # 메모리 부족 위험
```

#### 3. 디스크 I/O 최적화

```python
# 좋은 예: 스트림 쓰기
with open('results.jsonl', 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')

# 나쁜 예: 메모리에 모든 결과 저장 후 한번에 쓰기
all_results = []  # 메모리 사용량 증가
```

## 📈 모니터링과 분석

### 1. 실행 중 모니터링

```python
# 진행률 추적
from tqdm import tqdm

with tqdm(total=len(data), desc="진행률") as pbar:
    for item in data:
        result = process_item(item)
        pbar.update(1)
        pbar.set_postfix({
            'success_rate': f"{success_count/processed_count:.1%}",
            'avg_length': f"{avg_length:.1f}"
        })
```

### 2. 결과 분석

```python
# 결과 통계 대시보드
def create_analysis_report(results_file):
    df = pd.read_excel(results_file)
    
    report = {
        'summary': {
            'total_questions': len(df),
            'avg_length': df['question'].str.len().mean(),
            'unique_drugs': df['구분'].nunique()
        },
        'quality_metrics': {
            'pronoun_rate': check_pronoun_usage(df),
            'concrete_rate': check_concreteness(df),
            'label_distribution': df['라벨'].value_counts().to_dict()
        },
        'recommendations': generate_recommendations(df)
    }
    
    return report
```

## 🔄 지속적 개선 방법

### 1. A/B 테스트

```python
# 프롬프트 버전 테스트
def test_prompts(data_sample, prompt_a, prompt_b):
    results_a = generate_with_prompt(data_sample, prompt_a)
    results_b = generate_with_prompt(data_sample, prompt_b)
    
    metrics_a = evaluate_quality(results_a)
    metrics_b = evaluate_quality(results_b)
    
    winner = 'A' if metrics_a['overall_score'] > metrics_b['overall_score'] else 'B'
    return winner, metrics_a, metrics_b
```

### 2. 피드백 루프

```python
# 사용자 피드백 수집
def collect_feedback(questions):
    feedback = {}
    for q in questions:
        rating = input(f"질문: {q}\n평가 (1-5): ")
        feedback[q] = int(rating)
    return feedback

# 피드백 기반 개선
def improve_based_on_feedback(feedback):
    low_quality = [q for q, rating in feedback.items() if rating <= 2]
    analyze_failure_patterns(low_quality)
```

### 3. 버전 관리

```bash
# Git으로 프롬프트 버전 관리
git tag v5.1 -m "구체성 개선된 프롬프트"
git push origin v5.1

# 실험 브랜치
git checkout -b experiment/longer-questions
# 실험 후
git checkout main
git merge experiment/longer-questions
```

## 📚 추가 리소스

### 유용한 스크립트

1. **`utils/data_validator.py`**: 입력 데이터 검증
2. **`utils/quality_checker.py`**: 결과 품질 평가
3. **`utils/batch_processor.py`**: 대용량 데이터 배치 처리
4. **`utils/result_analyzer.py`**: 결과 분석 및 시각화

### 참고 문헌

- [OpenAI API 문서](https://platform.openai.com/docs)
- [Pandas 최적화 가이드](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- [비동기 프로그래밍 베스트 프랙티스](https://docs.python.org/3/library/asyncio.html)

### 커뮤니티

- **이슈 리포팅**: GitHub Issues
- **기능 요청**: GitHub Discussions
- **기술 질문**: Stack Overflow (태그: pharma-augment)

---

*최종 업데이트: 2025-09-03*  
*작성자: Claude Code Assistant*