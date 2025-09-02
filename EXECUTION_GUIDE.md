# 📋 실행 가이드

## 🎯 **현재 상태**

- ✅ **V1 (기본 QA)**: 완성 및 테스트 완료
- 🔄 **V2 (임베딩 최적화)**: 전체 실행 중 (예상 15-20분)

## 🚀 **실행 방법**

### **V1 실행 (일반 QA 학습용)**
```bash
cd v1_original
python generate_questions.py --excel "../data/요양심사약제_후처리_v2.xlsx" --out "questions_v1.jsonl"
```

### **V2 실행 (임베딩 모델 학습용)**

#### 방법 1: 배치 파일 사용 (Windows)
```cmd
cd v2_embedding_optimized
run_full_generation.bat
```

#### 방법 2: 직접 실행
```bash
cd v2_embedding_optimized  
python generate_questions_v2.py --excel "..\data\요양심사약제_후처리_v2.xlsx" --out "embedding_questions_v2.jsonl" --provider "openai" --model "gpt-4o-mini" --concurrency 6 --max_aug 20
```

## ⏰ **예상 시간**

| 버전 | 조항 수 | 예상 시간 | 예상 질문 수 |
|------|---------|-----------|--------------|
| V1 | 688개 | 15분 | ~10,000개 |
| V2 | 688개 | 15-20분 | ~15,000-20,000개 |

## 📊 **현재 진행 상황**

```
V2 전체 실행: 시작됨 (2025-09-02 18:13)
- 총 697개 항목 전처리 완료
- 예상 완료: 18:30-18:35
```

## 📁 **결과 파일**

### V1 출력
- `questions_v1.jsonl` - JSONL 원본
- `questions_v1_questions.xlsx` - 1행당 1질문
- `audit_log.csv` - 생성 로그

### V2 출력  
- `embedding_questions_v2.jsonl` - JSONL 원본
- `embedding_questions_v2_questions.xlsx` - 1행당 1질문
- `audit_log_v2.csv` - 생성 로그

## 🔍 **진행 상황 확인**

### 로그 실시간 모니터링
```bash
cd v2_embedding_optimized
tail -f question_generation_v2.log
```

### 파일 크기 확인
```bash
cd v2_embedding_optimized
dir *.jsonl
```

## ⚠️ **주의사항**

1. **API 키**: `.env` 파일에 올바른 OpenAI API 키 설정
2. **네트워크**: 안정적인 인터넷 연결 필요
3. **용량**: 결과 파일은 약 50-100MB 예상
4. **중단**: Ctrl+C로 중단 가능 (부분 결과 저장됨)

## 🎯 **최종 목표**

- **V1**: ChatGPT fine-tuning, RAG 시스템용
- **V2**: 임베딩 모델(Sentence-BERT, E5, BGE) 학습용

---

**현재 V2가 백그라운드에서 실행 중입니다. 15-20분 후 결과를 확인하세요!** ⏳