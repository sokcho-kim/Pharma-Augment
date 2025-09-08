# 🔍 V7 컬럼 매핑 문제 분석

## 🚨 문제 코드 (line 780-787)

```python
for q in all_questions:
    row_data = {
        '약제분류번호': q.metadata['code'],
        '약제 분류명': q.metadata['code_name'], 
        '구분': q.metadata.get('title', ''),           # ❌ 문제!
        '세부인정기준 및 방법': q.metadata.get('text', ''),  # ❌ 문제!
        'question': q.text,
        '라벨': q.label
    }
```

## 🔍 문제 원인

### 1. 잘못된 키 매핑
- **'구분' 컬럼**: `q.metadata.get('title', '')` 
- **'세부인정기준 및 방법' 컬럼**: `q.metadata.get('text', '')`

### 2. 실제 원본 데이터 구조 확인 필요
**원본 엑셀**: `C:\Jimin\Pharma-Augment\data\요양심사약제_후처리_v2.xlsx`
- 약제분류번호
- 약제 분류명  
- **구분** ← 이 컬럼의 실제 키는?
- **세부인정기준 및 방법** ← 이 컬럼의 실제 키는?

### 3. metadata 생성 지점 확인 필요
Question 객체 생성시 metadata에 올바른 키로 저장되었는지 확인

## 🔧 수정 방향

### 1. 원본 데이터 컬럼명 확인
```python
df = pd.read_excel("data/요양심사약제_후처리_v2.xlsx")
print(df.columns.tolist())
```

### 2. Question 생성 지점에서 metadata 확인
```python
# generate_questions_for_row 함수 내부
question = Question(
    text=text,
    label=label,
    band=band,
    anchor_id=anchor_id,
    doc_slice_id=idx,
    metadata={
        'code': row['???'],           # 실제 컬럼명
        'code_name': row['???'],      # 실제 컬럼명
        'title': row['???'],          # 구분 → 실제 키?
        'text': row['???'],           # 세부인정기준 → 실제 키?
    }
)
```

### 3. 정확한 매핑 수정
```python
row_data = {
    '약제분류번호': q.metadata.get('code', ''),
    '약제 분류명': q.metadata.get('code_name', ''), 
    '구분': q.metadata.get('구분', ''),                    # 정확한 키
    '세부인정기준 및 방법': q.metadata.get('세부인정기준 및 방법', ''), # 정확한 키
    'question': q.text,
    '라벨': q.label
}
```

## ⚠️ 내일 첫 번째 작업
1. 원본 데이터 컬럼 구조 확인
2. metadata 키 매핑 수정  
3. 1-2행 테스트로 검증
4. 사용자 확인 후 전체 실행

**절대 검증 없이 장시간 실행 금지!**