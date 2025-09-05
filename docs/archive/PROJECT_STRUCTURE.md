# Pharma-Augment 프로젝트 구조

의료보험 도메인 질문 생성 및 데이터 증강 프로젝트입니다.

## 📁 **폴더 구조**

```
C:\Jimin\Pharma-Augment\
├── 📁 data/                              # 원본 데이터
│   ├── 요양심사약제_후처리_v2.xlsx        # 메인 데이터 (688개 조항)
│   └── sample_요양심사약제_후처리_v2.xlsx # 테스트용 샘플 (10개 조항)
│
├── 📁 v1_original/                       # V1 (기본 질문 생성기)
│   ├── generate_questions.py            # 메인 생성 스크립트
│   ├── jsonl_to_excel.py               # JSONL→엑셀 변환기
│   ├── requirements.txt                 # 필수 패키지
│   ├── install_packages.bat/.sh         # 설치 스크립트
│   ├── prompt.md                        # V1 프롬프트 전략
│   └── README.md                        # V1 사용법
│
├── 📁 v2_embedding_optimized/           # V2 (임베딩 모델 최적화)
│   ├── generate_questions_v2.py        # 향상된 생성 스크립트
│   ├── requirements.txt                 # 필수 패키지
│   ├── prompt2.md                       # V2 프롬프트 전략
│   ├── README_v2.md                     # V2 사용법
│   ├── run_sample_test.bat              # 샘플 테스트 스크립트
│   ├── run_full_generation.bat          # 전체 실행 스크립트
│   └── .env                            # API 키 설정
│
├── 📄 PROJECT_STRUCTURE.md              # 이 파일
├── 📄 prompt.md                         # V1 전략 문서
└── 📄 prompt2.md                        # V2 전략 문서
```

## 🎯 **버전별 특징**

### **V1 - 기본 질문 생성기**
- **목적**: 일반적인 QA 학습 데이터 생성
- **질문 수**: 조항당 5~15개
- **전략**: 기본 5문형 + 증강
- **소요 시간**: 688개 조항 기준 15분
- **결과**: ~10,000개 질문

### **V2 - 임베딩 모델 최적화**
- **목적**: 벡터 검색 성능 향상을 위한 임베딩 모델 학습
- **질문 수**: 조항당 15~25개
- **전략**: 약제명/품명 균형 + 6가지 질문 유형
- **소요 시간**: 688개 조항 기준 15-20분
- **결과**: ~15,000-20,000개 질문

## 🚀 **사용 가이드**

### **V1 사용 (기본 QA 데이터)**
```bash
cd v1_original
python generate_questions.py --excel "../data/요양심사약제_후처리_v2.xlsx"
```

### **V2 사용 (임베딩 최적화)**
```bash
cd v2_embedding_optimized
run_sample_test.bat           # 테스트용
run_full_generation.bat       # 전체 실행
```

## 📊 **성능 비교**

| 항목 | V1 | V2 |
|------|----|----|
| 질문 생성 수 | 10K | 15-20K |
| 약제명 전략 | 기본 | 균형적 (30:30:25:15) |
| 질문 유형 | 5가지 | 6가지 |
| 벡터 검색 최적화 | ❌ | ✅ |
| 실무 시나리오 | 기본 | 강화 |

## 🔧 **개발 환경**

### **공통 요구사항**
- Python 3.8+
- OpenAI API 키
- 메모리: 8GB+ 권장
- 저장공간: 500MB+ 여유분

### **패키지 의존성**
```
pandas>=2.0.0
openpyxl>=3.1.0
aiohttp>=3.8.0
backoff>=2.2.0
rapidfuzz>=3.0.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
```

## 📈 **결과 파일**

### **V1 출력**
- `questions.jsonl` - 기본 JSONL 형식
- `questions_questions.xlsx` - 1행당 1질문 엑셀
- `audit_log.csv` - 생성 과정 로그

### **V2 출력**
- `embedding_questions_v2.jsonl` - 임베딩 최적화 JSONL
- `embedding_questions_v2_questions.xlsx` - 1행당 1질문 엑셀
- `audit_log_v2.csv` - V2 전용 감사 로그

## 🎓 **활용 방안**

### **V1 데이터 → 일반 QA 시스템**
```python
# ChatGPT fine-tuning, RAG 시스템 등
{"question": "질문", "answer": "답변", "context": "원문"}
```

### **V2 데이터 → 임베딩 모델 학습**
```python
# Sentence-BERT, E5, BGE 등 임베딩 모델 학습
{"query": "질문", "positive": "관련문서", "hard_negative": "유사하지만다른문서"}
```

---

**의료보험 도메인의 최고 품질 QA 데이터 생성 완료!** 🎯