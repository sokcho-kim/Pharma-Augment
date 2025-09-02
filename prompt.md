역할: 너는 한국 요양급여 심사 도메인의 “질문만 생성 + 증강” 배치 파이프라인을 만드는 코딩 에이전트다.
입력 데이터(확정): 요양심사약제_후처리_v2.xlsx / Sheet1 / 컬럼 = 약제분류번호, 약제분류명, 구분, 세부인정기준 및 방법
목표 산출물: 조항(row)당 질문만(답변·근거 금지) 최소 5개 + 증강 5~15개(중복 제거 후), JSONL로 저장.

0) 설치 & 실행 예시

설치:

pip install pandas openpyxl aiohttp backoff rapidfuzz python-dotenv tiktoken


실행 예시:

python generate_questions.py \
  --excel "요양심사약제_후처리_v2.xlsx" \
  --sheet "Sheet1" \
  --out "questions.jsonl" \
  --provider "openai" \
  --model "gpt-4o-mini" \
  --concurrency 6 \
  --max_aug 15 \
  --seed 20250902

1) 스크립트 스펙: generate_questions.py
1.1 CLI 옵션

--excel(필수), --sheet(기본 Sheet1), --out(기본 questions.jsonl)

--provider = openai|claude (기본 openai)

--model = 문자열 (기본 gpt-4o-mini)

--concurrency(기본 6), --max_aug(기본 15), --base5_only(기본 False)

--seed(기본 고정값), --neg_ratio(기본 0.0; 하드네거티브 라벨만 부여하고 질문은 최종 포함하지 않음)

--print-sample N (조항 N개 샘플 질문 콘솔 출력)

1.2 컬럼 고정 매핑(이 파일 전용)

clause_id: 아래 규칙으로 생성

title: 구분

text: 세부인정기준 및 방법

code: 약제분류번호(결측 가능)

code_name: 약제분류명(결측 가능)

추가로 title_clean = 구분에서 선행 태그(예: [일반원칙]) 제거한 문자열
category = 구분에 대괄호가 있을 경우 내부 텍스트(예: 일반원칙), 없으면 None

1.3 헤더 자동 매핑(견고성 확보)

본 파일은 위 컬럼명이 확정이지만, 견고성을 위해 아래 스코어링으로 자동 매핑도 구현:

clause_id 후보 키워드: 코드, 식별자, 번호, ID (+2), 정규식 ^[A-Za-z0-9\-_./]+$ 비율↑(+2), 유니크율↑(+2)

title 후보 키워드: 구분, 제목, 항목명, 조항명(+3), 평균 길이 5~40(+1)

text 후보 키워드: 세부인정기준, 내용, 본문, 지침, 기준, 설명(+3), 평균 길이≥120(+2), 줄바꿈/문장부호 많음(+1)

최종 다수결·스코어 최상 매핑. 충돌 시: text=최장, id=유니크율 최상, title=중간 길이.

1.4 clause_id 생성 규칙(결정적·재현성)
id_part = 약제분류번호(공백 제거) 가 있으면 그 값, 없으면 빈값
title_slug = 구분(또는 title_clean)을 소문자-한글자모 유지, 공백→하이픈, 괄호/기호 제거 후 40자 이하
hash8 = SHA1(세부인정기준 및 방법 원문).hexdigest()[:8]

if id_part:
    clause_id = f"{id_part}_{title_slug}"
else:
    clause_id = f"{title_slug}_{hash8}"

group_id = id_part if id_part else title_slug   # 슬라이스/버저닝 묶음용

1.5 긴 조항 슬라이싱

text 길이 > 6000자 → 문단 경계 기준 2500~3000자 chunk로 분할

분할 시 clause_id에 _p1,_p2,... 부여, group_id로 원본 묶음 유지

1.6 LLM 호출 (기본 OpenAI, 선택 Claude)

OpenAI: chat.completions, model(기본 gpt-4o-mini), temperature=0.5, top_p=0.9, response_format={"type":"json_object"}, max_tokens는 조항 길이에 따라 800~1200

Claude: Messages API, system+user 프롬프트, JSON만 출력하도록 강제

공통: 429/5xx/JSON위반 시 백오프 재시도 1회(지수형), 실패 시 로그 기록 후 건너뜀

2) “질문 전용” 생성 규칙
2.1 프롬프트(모델 공통)
[역할] 너는 한국 요양급여 심사 도메인의 '질문 생성 에이전트'다.
[제약] '질문만 생성'. 답변/근거/해설 절대 금지. 외부 지식/추정 금지.
[입력]
- clause_id: {clause_id}
- title: {title}
- title_clean: {title_clean}
- category: {category}   # 없으면 null
- code: {code}           # 없으면 null
- code_name: {code_name} # 없으면 null
- text: """
{text}
"""

[출력 스키마]
{
  "questions": ["문장1","문장2","..."]   // JSON 객체 하나만
}

[생성 규칙]
1) '기본 5문형'을 먼저 생성:
   - 정의/범위, 요건/기준, 제외/불인정, 증빙/서류, 엣지/경계(재시술/합병증/동반상병/진료장소 등 '원문에 언급된 경우에만')
2) 이어서 '증강 질문' 5~{max_aug}개:
   - WH 다양화(무엇/언제/어디서/누가/어떤 조건/어떻게/왜)
   - 문체 변형(~인가요/~해야 하나요/~가능한가요/~허용되나요/~인정되나요)
   - 길이 변형(짧은 요점형 ↔ 구체 조건형)
   - 주체/시점 명시(신청자/의료기관/담당부서, 최초/재시술/추적관찰)
   - 조건 조합(~인 경우에도 인정되나요? / ~이면 제외되나요?)
3) 원문에 없는 개념·수치·기관명·연도 등 삽입 금지.
4) 각 질문은 1문장, 한국어, 길이 15~180자.
5) JSON 외 출력 금지.

2.2 사후 처리(포스트프로세싱)

정규화: 공백/중복기호/말줄임 보정, 문장부호 최종 ? 권장

중복 제거: rapidfuzz 토큰세트비율 ≥ 90 → 하나만 남김

금지어 필터: 추정, 일반적으로, 대체로, 관행상, 아마도 포함 시 제거

외부지식 의심: 타 기관/연도/외부수가 레퍼런스 단어 등장 시 제거

길이 필터: 15~180자 유지

결과가 5개 미만이면 **한 번 더 생성 요청(증강만)**하여 충족

3) 출력 스키마(최종 JSONL)

줄당 1 조항 객체

{
  "clause_id": "H1234_p1",
  "group_id": "H1234",
  "title": "구분 컬럼 원문",
  "title_clean": "대괄호 태그 제거 버전",
  "category": "일반원칙",
  "code": "약제분류번호 또는 null",
  "code_name": "약제분류명 또는 null",
  "questions": [
    "이 조항의 적용 범위는 무엇인가요?",
    "급여 인정 요건은 무엇인가요?",
    "다음과 같은 상태에서는 제외되나요? (원문에 언급된 경우만)",
    "청구 시 제출해야 하는 서류는 무엇인가요?",
    "재시술 또는 동반상병이 있는 경우 적용이 달라지나요?"
    // + 증강문항(중복 제거 후)
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

4) 감사(audit) 로그

audit_log.csv: clause_id, num_questions, retries, provider, model, tokens_req, tokens_resp, elapsed_ms

5) 성능/안정화 팁

동시성 6~8, 지수 백오프

긴 조항(>6k자)은 분할 후 슬라이스별 생성 → 병합 시 중복 제거

temperature=0.5, top_p=0.9(증강 다양성), 응답은 JSON만 요구하여 토큰 절약

샘플 출력: --print-sample 5로 무작위 5개 조항 질문 프린트

6) 유닛 테스트(간단)

샘플 10행에 대해:

questions 길이 ≥ 5

금지어/외부지식 키워드 없음

평균 길이 15~180자 범위

JSONL 파싱 100% 성공

🔧 모델 추천 (gpt 계열)

생성 기본: gpt-4o-mini (빠르고 저렴, JSON 안정)

temperature=0.5, top_p=0.9, response_format={"type":"json_object"}

재시도/교정 전용(옵션): gpt-4o

긴 조항/어휘 난도 높을 때 실패건 재생성에만 부분 투입

📌 이 파일 전용 매핑 결론(확정)

title ← 구분

text ← 세부인정기준 및 방법

code / code_name ← 약제분류번호 / 약제분류명(결측 허용)

clause_id ← 위 규칙(있으면 약제분류번호_구분-slug, 없으면 구분-slug_hash8)

구분 앞의 [일반원칙] 등 태그는 category로 추출, title_clean은 태그 제거본 사용