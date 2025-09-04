# Pharma-Augment V5 - 향상된 질문 생성기

## 🎯 V5 특징

**V2의 풍부함 + V4의 엄격함을 결합한 최고 품질 버전**

- ✅ **V2 수준의 구체적 질문**: 25-80자, 의료용어 포함
- ✅ **대명사 완전 차단**: "이것", "해당 약제" 등 일체 금지
- ✅ **품질 중심 필터링**: 구체성, 실무성 검증
- ✅ **5가지 질문 유형**: 정보/조건/비교/절차/제한형
- ✅ **창의성 증가**: temperature 0.8

## 📊 예상 성능

| 항목 | V5 목표 |
|------|---------|
| 평균 질문 길이 | 30-50자 |
| 성공률 | 95% 이상 |
| 대명사 사용률 | 0% |
| 품질 점수 | V2 수준 |

## 🚀 실행 방법

### 기본 기능 테스트 (API 키 불필요)
```bash
# Windows
run_v5_basic.bat

# Linux/Mac
cd versions/v5
python test_v5_basic.py
```

### Mock API 테스트 (API 키 불필요)
```bash
# Windows  
run_v5_mock.bat

# Linux/Mac
cd versions/v5
python test_v5_mock.py
```

### 샘플 테스트 (API 키 필요)
```bash
# Windows
run_v5_test.bat

# Linux/Mac
cd versions/v5
python drug_generator_v5.py \
  --excel "../../data/sample_요양심사약제_후처리_v2.xlsx" \
  --out "test_drug_questions_v5.xlsx" \
  --concurrency 2
```

### 전체 실행 (API 키 필요)
```bash
# Windows
run_v5_full.bat

# Linux/Mac  
cd versions/v5
python drug_generator_v5.py \
  --excel "../../data/요양심사약제_후처리_v2_cleaned.xlsx" \
  --out "drug_questions_v5_final.xlsx" \
  --concurrency 6
```

## 📝 명령행 옵션

```bash
python drug_generator_v5.py [옵션]

필수:
  --excel PATH         입력 엑셀 파일 경로

선택:
  --sheet SHEET        시트명 (기본: Sheet1)
  --out FILE           출력 파일명 (기본: drug_questions_v5.xlsx)
  --provider PROVIDER  LLM 제공자 (openai/claude)
  --model MODEL        모델명 (기본: gpt-4o-mini)
  --concurrency N      동시 실행 수 (기본: 6)
  --seed N             랜덤 시드 (기본: 20250903)
```

## 🔍 V5 품질 기준

### 1. 길이 검증
- **25-80자**: V2 수준의 구체성 확보
- 너무 짧거나 긴 질문 자동 필터링

### 2. 대명사 차단
```python
# 차단되는 패턴
"(이|그|해당|본|동)\s?(약|약제|제제|제품)|이것|그것"
```

### 3. 구체성 검증
- 숫자/단위 포함 필수
- 의료용어 활용 필수  
- 구체적 명사 사용 필수

### 4. 질문 유형 다양화
- A) 정보형: "구체적인 급여 인정 기준은?"
- B) 조건형: "특정 수치일 때 사용 가능한가요?"
- C) 비교형: "경구제와 주사제 차이점은?"
- D) 절차형: "필요한 사전승인 절차는?"
- E) 제한형: "사용이 제한되는 경우는?"

## 📈 출력 형식

### 엑셀 출력 (고정 컬럼)
```
약제분류번호 | 약제 분류명 | 구분 | 세부인정기준 및 방법 | question | 라벨
```

### 감사 로그
```csv
row_id, row_idx, main_name, brand_count, questions_generated, avg_length, elapsed_ms, provider, model, version
```

## 🎯 V5 vs 다른 버전

| 버전 | 평균 길이 | 품질 | 커버리지 | 대명사 차단 |
|------|-----------|------|----------|-------------|
| V2 | 36.2자 | ⭐⭐⭐⭐⭐ | 85% | ❌ |
| V3 | ~30자 | ⭐⭐⭐⭐ | 70% | ✅ |
| V4 | ~25자 | ⭐⭐ | 100% | ✅ |
| **V5** | **30-50자** | **⭐⭐⭐⭐⭐** | **95%** | **✅** |

## 🔧 환경 설정

### .env 파일
```bash
OPENAI_API_KEY=your_api_key_here
# 또는
ANTHROPIC_API_KEY=your_claude_key_here
```

### 의존성 설치
```bash
pip install -r ../../requirements.txt
```

## 📊 예상 결과

**V5로 생성한 질문 예시:**
```
✅ "AST 수치가 60U/L 이상일 때 Tacrolimus 제제 급여요건은 어떻게 적용되나요?" (42자)
✅ "프로그랍캅셀을 3개월 이상 장기 처방 시 필요한 모니터링 항목은?" (35자)
✅ "간기능 저하 환자에서 Tacrolimus 제제 용량 조절 기준과 주의사항은?" (37자)
```

## 🚨 주의사항

1. **API 키 필수**: OpenAI 또는 Anthropic API 키 설정 필요
2. **메모리 사용량**: 대용량 데이터 처리시 8GB+ 권장
3. **네트워크**: 안정적인 인터넷 연결 필요
4. **속도**: 688행 기준 약 15-20분 소요

## 📞 지원

- V5 실행 오류 시 로그 파일 확인: `drug_generation_v5.log`
- 감사 로그로 상세 분석: `audit_log_drug_v5.csv`

---

*V5는 Pharma-Augment의 최신 버전으로, 최고 품질의 의료보험 질문 데이터를 생성합니다.*