# Pharma-Augment

한국 요양급여 심사 도메인 질문 생성 및 데이터 증강 프로젝트

## 프로젝트 구조

```
Pharma-Augment/
├── versions/           # 버전별 구현체
│   ├── v1/            # 기본 질문 생성 (기존)
│   ├── v2/            # 임베딩 최적화 버전
│   └── v3/            # 대명사 차단 + 이름비율 강제 (최신)
├── data/
│   ├── inputs/        # 입력 엑셀 파일
│   └── outputs/       # 생성 결과 파일
├── utils/             # 유틸리티 스크립트
├── docs/              # 문서 및 프롬프트
└── logs/              # 실행 로그
```

## 버전별 특징

### V1 (기본)
- 기본적인 질문 생성
- WH 질문 중심
- 길이/품질 필터링

### V2 (임베딩 최적화)
- 임베딩 학습 최적화
- 의미 다양성 강화
- 부정/긍정 균형

### V3 (최신 - Prompt3.md 기반)
- **대명사 완전 차단**: "이 약제", "해당 제제" 등 금지
- **이름 사용 비율 강제**: MAIN/BRAND/BOTH 비율 엄격 준수
- **9개 카테고리 분산**: 범위/요건/오프라벨/기간/전환/증빙/본인부담/대상군/절차
- **브랜드명 개수별 비율 자동 조정**

## 실행 방법

### V3 실행 (권장)
```bash
python versions/v3/generate_questions_v3.py --excel data/inputs/요양심사약제.xlsx --out data/outputs/questions_v3.jsonl --excel_out data/outputs/questions_v3.xlsx
```

### 환경 설정
```bash
# 패키지 설치
python -m pip install -r requirements.txt

# 환경변수 설정 (.env 파일)
OPENAI_API_KEY=your_api_key_here
```

## 주요 특징

- **엄격한 검증 시스템**: 대명사 정규식 체크, 이름 사용 검증, 비율 검증
- **약제별 맞춤 생성**: 브랜드명 개수에 따른 비율 자동 조정  
- **다양한 출력 형식**: JSONL, 엑셀 지원
- **상세한 감사 로그**: API 호출, 검증 결과 추적

## 출력 형식

```json
{
  "drug_id": "A01AD02_tacrolimus",
  "main_name": "Tacrolimus 제제", 
  "brand_names": ["프로그랍캅셀", "프로그랍주사"],
  "questions": [
    {"text": "Tacrolimus 제제의 조혈모세포이식 급여 범위는?", "name_usage": "MAIN", "category": "범위"},
    {"text": "프로그랍캅셀의 만성 류마티스관절염 인정 기준은?", "name_usage": "BRAND", "category": "요건"}
  ],
  "ratio": {"MAIN": 0.35, "BRAND": 0.35, "BOTH": 0.30}
}
```