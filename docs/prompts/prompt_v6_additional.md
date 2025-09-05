✅ 운영 지시문 (for Claude Code)
0) 목표

임베딩 파인튜닝용 질문 데이터 생성.

길이 밴드 혼합으로 키워드 과적합 방지:
SR(25–80자) 60% · MR(80–160자) 25% · LR(200–600자, 시나리오) 15%.

라벨 비율(행 단위): POS : HN : EN = 6 : 3 : 0(기본). EN은 실험시에만 on.

약제(Drug) / 고시(Notice) 파이프라인 완전 분리. 공통 유틸만 공유.

1) 입·출력 스키마
입력 엑셀 헤더 매핑

약제: 약제분류번호→code, 약제 분류명→code_name, 구분→title, 세부인정기준 및 방법→text

고시: 고시번호→code, 고시명칭→code_name, 변경 전 내용→text_prev, 변경 후 내용→text

제출용 출력(엑셀)

고정 열: [약제분류번호, 약제 분류명, 구분, 세부인정기준 및 방법, question, 라벨]

약제: 그대로 매핑.

고시: 약제분류번호=고시번호, 약제 분류명=고시명칭, 구분=고시명칭, 세부인정기준 및 방법=변경 후 내용.

2) 전처리

컬럼 매핑 후 결측 체크(필수 열 없으면 해당 행 skip 로그).

문단 경계 기준 2–3k자 슬라이싱(중간 문장 끊지 말 것).

약제 전용 파싱(title):

main_name=괄호 앞 주 약제명,

brand_names="품명:" 뒤를 ·// split(없으면 []).

정답 문서(Positive 근거) 선택:

약제: text 슬라이스.

고시: 기본 text(=변경 후 내용); 질문에 “개정 전/시행 전”이 명시되면 text_prev.

3) 생성(LLM) — 줄단위 텍스트만, 프롬프트는 영어 / 출력은 한국어

공통 제약(생성기에게만):

One line = one question. Korean only. No JSON.

Pronoun ban: forbid 이것/그것/해당/본/동 + (약|제제|제품|고시|조항|내용|항).

Include at least one of: digit/unit/policy term
(policy: 급여, 비급여, 본인부담, 사전승인, 수가, 코드, 기간, 횟수, 시행일, 개정)

Vary openings (“무엇/어떻게/언제/왜/어떤/어디서/어느/누가”) so no single token > 30% in a set.

No external inference; use the document only.

3-A) POS 생성 — 약제 (SR/MR/LR 각각 호출)

SR (25–80 chars)

You generate Korean questions for an insurance review embedding model.
Return many lines, each a single Korean question strictly based on the document below.
Constraints: 25–80 characters; end with '?'; include at least one number/unit/policy term;
no pronouns like '이것/그것/해당/본/동 + 약·제제·제품'; one issue per line; vary openings; no JSON.

Document:
<<<DOC
{drug_text_slice}
DOC
>>>


MR (80–160 chars)

Generate many Korean questions as lines based only on the document.
Constraints: 80–160 characters; end with '?'; include at least one number/unit/policy term;
strict pronoun ban; one issue per line; vary openings; no JSON.

Document:
<<<DOC
{drug_text_slice}
DOC
>>>


LR (200–600 chars, scenario → question)

Create multiple Korean scenarios (2–4 sentences) followed by a final question line.
Separate each case with a blank line. Use only the document content.
Constraints: 200–600 characters per case; the last sentence must be a question ending with '?';
include at least one policy term or number/unit; avoid pronouns in the question; no JSON.

Document:
<<<DOC
{drug_text_slice}
DOC
>>>


약제 문구 힌트(프롬프트에 넣지 말고 코드에서 참고): 급여/비급여/기간/횟수/경구↔주사/사전승인/본인부담/대상군/제출서류/모니터링.

3-B) POS 생성 — 고시 (SR/MR/LR)

SR/MR는 위 약제 프롬프트에서 Document만 notice_after_text_slice로 교체.
LR도 동일하되 “수가/코드/시행일/개정/제출/이의신청” 용어가 자연스럽게 포함되도록 문서 내용만 사용.

4) HN(near-miss) 생성 — 코드 변형 + 리라이트

원칙: “동일 성분/핵심 키워드 유지 + facet 1개만 의도적으로 어긋남”.

변형은 코드가 수행(LLM은 리라이트만).

4-A) 변형 규칙(코드)

고정: 성분/주 약제명(+가능하면 제형/경로), 품명/코드 토큰.

변경(facet 1개만): 적응증, 대상/연령, 기간/횟수 경계, 용량/간격, 경구↔주사, 급여↔본인부담, 사전승인, 선행요법, 모니터링.

고시 전용: 변경 전↔후 혼동, 초진↔재진, 시행일·적용일, 제출서류 유무, 기간/횟수 상한 경계.

앵커 선택: POS 상위 3–5개. 각 앵커당 HN 1–3개.

4-B) HN 리라이트 프롬프트(공통)
Rewrite the given Korean sentence into a natural Korean question.
Constraints: 25–80 characters; end with '?'; keep numbers/units/policy terms; no pronouns.
Return exactly one line. No JSON.

Original:
{mutated_anchor_sentence}

5) 후처리 게이트(강제)

길이 밴드

SR: 25–80자, MR: 80–160자, LR case: 200–600자(마지막 문장만 question으로 추출).

문장부호: ? 필수.

대명사 금지(정규식):
(이것|그것|해당|본|동)\s*(약|제제|제품|고시|조항|내용|항)|\b(이것|그것)\b

LR 시나리오는 본문 서술에 한해 허용하되 질문문에는 금지.

구체성: 숫자/단위 (mg|㎎|U/L|%|회|개월|일|주) 또는 정책어(위 목록) ≥1.

단일 논점: ,|및|/ 2회 이상이면 분리/폐기.

문두 분포: 동일 문두 >30% → 초과분 재프레이즈 큐.

본문 오버랩: 스톱워드 제외 교집합 비율 ≥0.25.

중복 제거: RapidFuzz token_set_ratio ≥ 82 or 4/5-gram 중복 → 제거.

HN 검증(룰 체커): 고정 토큰 포함 + facet 정확히 1개만 변했는지 확인(2개 이상이면 폐기/강등).

6) 라벨 비율 정규화(행 단위)

목표 N개, weights = [POS 6, HN 3, EN 0](합=9).

각 라벨 목표 = floor(N * w/9), 남는 개수는 소수점 큰 순으로 POS→HN→EN 배분.

기본은 EN=0. 실험시만 EN on.

7) 길이 밴드 비율(라벨과 별개)

SR 60% · MR 25% · LR 15%.

각 밴드 내부에서도 라벨 비율(6:3:0)을 대략 유지(±1).

8) 실패·견고화

생성 줄 수 부족(밴드별 <10줄): temperature↑ 후 2회 재시도.

429/timeout: 지수 백오프(2→4→8→최대 20s), 최대 3회.

대명사/길이/문두 과다 위반: 해당 부분 부분 재생성.

brand_names 없음(약제): POS는 주 약제명 중심, HN은 facet 변형 위주.

9) 저장

각 질문을 한 행으로 확장, 원본 메타 복사.

엑셀 저장(UTF-8): [약제분류번호, 약제 분류명, 구분, 세부인정기준 및 방법, question, 라벨].

(선택) 학습용 JSONL 앵커팩도 병행 저장:
{"anchor_id", "band": "SR/MR/LR", "question", "doc_slice_id", "label"}
— 같은 행/슬라이스의 SR·MR·LR을 동일 anchor_id로 묶어 사용.

10) 옵션(A/B 실험)

비율 비교: 6:3:0 vs 6:3:1(EN 10%) — 소규모 생성 후 Dev Recall@10 / nDCG@10 비교.

고시 “개정 전” 전용 슬라이스 세트 별도 출력(리스크 점검용).

🧩 구현 메모(내부)

생성기는 최소 지시(간결 프롬프트), 중요 규칙은 코드/체커에서 보증 → 패턴 고착/프롬프트 게이밍 방지.

HN은 항상 코드가 facet 1개만 바꾸고 LLM은 리라이트만.

LR 시나리오는 마지막 문장을 question으로 추출, 나머지 문장은 학습시 멀티뷰(앵커팩)로 활용 가능.