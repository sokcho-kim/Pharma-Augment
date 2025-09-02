# 의료 보험 임베딩 모델용 향상된 질문 생성 전략

## 1. 주 약제명과 품명 균형 전략

### 기본 원칙
- 각 약제당 생성되는 질문의 30-40%는 주 약제명 사용
- 30-40%는 품명 사용
- 20-30%는 둘 다 사용
- 10%는 대명사나 간접 지칭

### 예시 (Tacrolimus 기준)
```python
def generate_balanced_questions(drug_info):
    """
    drug_info = {
        'main_name': 'Tacrolimus 제제',
        'brand_names': ['프로그랍캅셀', '프로그랍주사'],
        'content': '세부인정기준 및 방법...'
    }
    """
    questions = []
    
    # 패턴 1: 주 약제명만 사용 (30-40%)
    questions.extend([
        "Tacrolimus 제제의 요양급여 범위는 무엇인가요?",
        "Tacrolimus를 조혈모세포이식에 사용할 때 기준은?",
        "Tacrolimus 투여 시 필요한 증빙 서류는?"
    ])
    
    # 패턴 2: 품명만 사용 (30-40%)
    questions.extend([
        "프로그랍캅셀의 급여 인정 기준은 무엇인가요?",
        "프로그랍주사를 사용할 수 있는 환자군은?",
        "프로그랍 제제의 용법용량 기준은?"
    ])
    
    # 패턴 3: 둘 다 사용 (20-30%)
    questions.extend([
        "Tacrolimus(프로그랍캅셀)의 투여 기간 제한은?",
        "프로그랍주사(Tacrolimus)의 병용 금기 약물은?",
        "Tacrolimus 제제인 프로그랍의 삭감 예외 사항은?"
    ])
    
    # 패턴 4: 간접 지칭 (10%)
    questions.extend([
        "이 면역억제제의 소아 사용 기준은?",
        "해당 약제의 신장이식 후 용량은?"
    ])
    
    return questions
```

## 2. 질문 다양성 증강 전략

### 2.1 질문 유형별 템플릿

#### A. 기본 정보형 (What)
- "{약제명}의 요양급여 인정 기준은?"
- "{품명}은 어떤 질환에 사용 가능한가요?"
- "{약제명}({품명})의 적응증은 무엇인가요?"

#### B. 조건/상황형 (When/If)
- "{특정상황}에서 {약제명}을 사용할 때 급여 기준은?"
- "{품명}을 {환자군}에게 투여 시 제한사항은?"
- "만약 {조건}이라면 {약제명} 사용이 가능한가요?"

#### C. 비교형 (Compare)
- "{약제명}의 경구제와 주사제 급여 기준 차이는?"
- "{품명}과 다른 {동일계열약}의 인정 기준 비교는?"
- "성인과 소아에서 {약제명} 사용 기준의 차이점은?"

#### D. 부정형 (Not/Exclude)
- "{품명}이 급여 인정되지 않는 경우는?"
- "{약제명}의 사용이 제한되는 환자군은?"
- "어떤 경우에 {품명} 삭감이 발생하나요?"

#### E. 절차형 (How/Process)
- "{약제명} 사용을 위한 사전승인 절차는?"
- "{품명} 처방 시 필요한 검사 항목은?"
- "어떻게 {약제명}의 급여를 신청하나요?"

### 2.2 난이도 계층화

#### 초급 (단순 사실 확인)
```python
easy_questions = [
    "프로그랍캅셀의 보험 코드는?",
    "Tacrolimus의 일일 최대 용량은?",
    "이 약제의 급여 적용 시작일은?"
]
```

#### 중급 (조건 해석)
```python
medium_questions = [
    "AST 수치가 100일 때 Tacrolimus 사용 가능 여부는?",
    "프로그랍을 3개월 이상 사용 시 필요한 모니터링은?",
    "신기능 저하 환자에서 용량 조절 기준은?"
]
```

#### 고급 (복합 판단)
```python
hard_questions = [
    "다제내성 결핵 환자에서 면역억제제인 Tacrolimus와 항결핵제 병용 시 급여 인정 기준과 주의사항은?",
    "소아 간이식 환자에서 프로그랍캅셀에서 프로그랍주사로 전환 시 용량 계산법과 급여 청구 방법은?",
    "Tacrolimus 혈중농도가 목표치 미달이면서 부작용 발생 시 급여 심사 기준은?"
]
```

## 3. 실무 시나리오 기반 질문

### 3.1 삭감 방어 관련
```python
audit_defense_questions = [
    "Tacrolimus 처방이 삭감된 경우 이의신청 근거는?",
    "프로그랍캅셀 장기처방 시 삭감 예방 방법은?",
    "비급여 전환을 피하기 위한 {약제명} 처방 전략은?"
]
```

### 3.2 청구 실무 관련
```python
billing_questions = [
    "프로그랍주사와 경구제 동시 처방 시 청구 방법은?",
    "Tacrolimus 약제비와 관련 검사료 동시 청구 기준은?",
    "입원과 외래에서 {품명} 청구 코드 차이는?"
]
```

## 4. 데이터 증강 실행 계획

### Phase 1: 기존 질문 리밸런싱
1. 각 약제별로 주 약제명/품명 사용 빈도 분석
2. 불균형한 질문들을 위 패턴으로 재작성
3. 카테고리당 최소 20개 질문 확보

### Phase 2: 다양성 증강
1. 5가지 질문 유형 × 3가지 난이도 = 15개 변형 생성
2. 부정형/비교형 질문 30% 이상 포함
3. 실무 시나리오 질문 20% 추가

### Phase 3: 품질 검증
1. 의료 전문가 검토 (샘플 100개)
2. 질문-답변 정합성 검증
3. 중복 제거 및 표현 다듬기

## 5. 예상 결과

### 증강 전
- 총 질문: 10,047개
- 평균 질문/카테고리: 14.6개
- 질문 유형: 7가지

### 증강 후 (목표)
- 총 질문: 20,000개+
- 평균 질문/카테고리: 30개
- 질문 유형: 15가지 (5유형 × 3난이도)
- 주약제명:품명 비율: 40:40:20

## 6. 구현 코드 예시

```python
import pandas as pd
from typing import List, Dict
import random

class MedicalQuestionAugmenter:
    def __init__(self, source_data: pd.DataFrame):
        self.source_data = source_data
        self.question_templates = self.load_templates()
        
    def augment_questions(self, row: Dict) -> List[str]:
        """각 행에 대해 증강된 질문 생성"""
        questions = []
        
        # 약제 정보 추출
        drug_info = self.extract_drug_info(row['구분'])
        content = row['세부인정기준 및 방법']
        
        # 1. 균형잡힌 약제명 사용
        questions.extend(self.generate_balanced_names(drug_info, content))
        
        # 2. 다양한 질문 유형
        questions.extend(self.generate_diverse_types(drug_info, content))
        
        # 3. 난이도별 질문
        questions.extend(self.generate_by_difficulty(drug_info, content))
        
        # 4. 실무 시나리오
        questions.extend(self.generate_practical_scenarios(drug_info, content))
        
        return questions
    
    def extract_drug_info(self, category: str) -> Dict:
        """카테고리에서 약제 정보 추출"""
        # 정규표현식으로 주 약제명과 품명 추출
        import re
        
        main_drug = re.match(r'^([^(]+)', category)
        brand_names = re.search(r'품명\s*:\s*([^)]+)', category)
        
        return {
            'main_name': main_drug.group(1).strip() if main_drug else '',
            'brand_names': [b.strip() for b in brand_names.group(1).split('·')] if brand_names else [],
            'category': category
        }
    
    def generate_balanced_names(self, drug_info: Dict, content: str) -> List[str]:
        """주 약제명과 품명을 균형있게 사용한 질문 생성"""
        questions = []
        
        # 구현 로직...
        
        return questions

# 사용 예시
augmenter = MedicalQuestionAugmenter(source_df)
augmented_df = source_df.apply(
    lambda row: augmenter.augment_questions(row), 
    axis=1
)
```

## 7. 검증 메트릭

### 정량적 지표
- 주약제명/품명 사용 비율: 목표 40:40:20 (±5%)
- 질문 유형 분포: 각 유형 최소 15%
- 난이도 분포: 초급 40%, 중급 40%, 고급 20%
- 중복률: < 1%

### 정성적 지표
- 의료 전문가 적절성 평가: > 90%
- 실무 활용도 평가: > 85%
- RAG 검색 성능 향상도: > 30%

## 8. 추가 고려사항

### 의료 용어 변형
- 약어 처리: "TAC" ↔ "Tacrolimus"
- 한글/영문 병기: "타크로리무스" 포함
- 오타 내성: "타크롤리무스", "프로그라프" 등

### 컨텍스트 확장
- 관련 검사명 포함 (예: "Tacrolimus 혈중농도 검사")
- 병용 약물 언급 (예: "스테로이드와 Tacrolimus 병용")
- 진료과 특화 (예: "신장내과에서 프로그랍 사용 시")