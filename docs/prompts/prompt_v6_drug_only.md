# 의료 보험 임베딩용 질문 생성 — 약제 전용 V6 (for Claude Code)

> 핵심 변화: **자유생성(텍스트만) → 강한 전·후처리**, **라벨 비율 POS 중심(6:3:0)**, **약제만 처리**.

---

## 0) TL;DR

* **LLM 출력 형식**: JSON 금지. **줄단위 질문 문장**만 대량 생성.
* **라벨 비율(행 단위)**: **POS : HN : EN = 6 : 3 : 0**(기본).
* **정답 문서(Positive 근거)**: **약제 `세부인정기준 및 방법` 슬라이스**
* **HN 원칙**: **동일 성분/핵심 키워드 유지 + facet 1개만 의도적으로 어긋나는 near-miss**
* **제출 엑셀 스키마**: `[약제분류번호, 약제 분류명, 구분, 세부인정기준 및 방법, question, 라벨]`

---

## 1) 입출력 규격

### 입력(엑셀 헤더 매핑)

* **약제(Drug)**
  `약제분류번호 → code` · `약제 분류명 → code_name` · `구분 → title` · `세부인정기준 및 방법 → text`

### 출력(제출용)

* **약제 스키마**
  `[약제분류번호, 약제 분류명, 구분, 세부인정기준 및 방법, question, 라벨]`
* **라벨**: POSITIVE | HARD_NEGATIVE

---

## 2) 전처리(반드시 수행)

1. **컬럼 매핑**: 한국어 헤더 부분일치로 표준 키로 rename.
2. **슬라이스**: 본문 길면 **문단 경계 기준 2~3k자**로 분할.
3. **메타 태깅(정규식)**: 숫자(횟수/개월/일), 단위(mg, U/L, %, 회…), 정책어(급여/비급여/사전승인/코드/수가/기간/본인부담).
4. **약제 전용 파싱(제목)**
   * `main_name` = 괄호 앞 주 약제명
   * `brand_names` = "품명:" 뒤를 `·` 또는 `/`로 분리(없으면 `[]`)
5. **정답 문서 선택**: 약제 `text` 슬라이스

---

## 3) 생성 프롬프트(텍스트만, 줄단위) — **프롬프트는 영어 / 출력은 한국어**

### 공통 규칙

* One line = one question, **25–80 Korean characters**, must end with `?`
* **No pronouns**: forbid `이것|그것|해당|본|동` + `(약|제제|제품)`
* **Specificity**: must include **at least one** of digit/unit/policy keyword
  (policy keywords: 급여/비급여/본인부담/사전승인/수가/코드/기간/횟수)
* **Vary openings**: do not let any single opening token ("무엇/어떻게/언제/왜/어떤/어디서/어느/누가") exceed **30%** within a set.
* **No external inference**: stick to the given text only.
* Output **Korean questions only**, one per line. **No JSON.**

### A) DRUG — POS generator prompt (EN)

```
You are generating Korean training questions for an insurance review embedding model.
Return many lines, each a single Korean question based ONLY on the document below.
Constraints:
- 25–80 characters, end with '?'
- Include at least one of: a number, a unit (mg, U/L, %, 회, 개월, 일, 주), or a policy term (급여, 비급여, 본인부담, 사전승인, 수가, 코드, 기간, 횟수)
- Strictly forbid pronouns like '이것/그것/해당/본/동 + 약·제제·제품'
- One issue per sentence, vary openings so no single opening exceeds 30% in a set
- Use main drug name and brand names if present; do NOT output JSON.

Document (use only this content):
<<<DOC
{drug_text_slice}
DOC
>>>
```

### B) DRUG — HN rewriter prompt (EN)

```
Rewrite the given Korean sentence into a natural Korean question.
Constraints: 25–80 characters, end with '?', keep any numbers/units/policy terms, no pronouns.
Return exactly one line with the question. No extra words, no JSON.

Original:
{mutated_anchor_sentence}
```

> `mutated_anchor_sentence`는 **코드**가 생성: 앵커 문장에서 **성분/핵심 토큰 고정**, **facet 1개만** 바꾼 문장(예: 적응증, 연령, 기간 경계, 경구↔주사, 사전승인, 급여/본인부담…).

---

## 4) 후처리(강제 게이트)

* **길이**: 25-80자, **물음표 필수**, **대명사 금지 정규식** 통과
  `r'(이것|그것|해당|본|동)\s*(약|제제|제품)|\b(이것|그것)\b'`
* **구체성 게이트**: 숫자 또는 단위 `(mg|U/L|%|회|개월|일|주|㎎)` 또는 정책어(급여/비급여/본인부담/사전승인/수가/코드/기간/횟수) **중 ≥1 포함**
* **단일 논점**: `,|및|/` **2회 이상**이면 분리/폐기
* **문두 분포 제어**: 동일 문두 >30% → 초과분 **재프레이즈 큐**로 이동
* **본문 오버랩**: 스톱워드 제외 핵심 토큰 교집합 비율 **≥0.25**
* **중복 제거**: RapidFuzz `token_set_ratio ≥ 82` 또는 4/5-gram 중복 → 제거
* **HN 검증(체커/룰)**
  * 앵커와 **핵심 키워드 공유(성분/코드/제형 등)**
  * facet 키워드 **≥1**, 그리고 **정확히 1개 차원만** 어긋남(두 개 이상 바뀌면 폐기/강등)
* **라벨 충족(행 단위)**: **POS 6 / HN 3 / EN 0** 목표(±1 허용). 부족 시 재생성.

---

## 5) 라벨 할당(6:3:0) — **정규화 규칙**

* 목표 N개에 대해 `weights = [6,3,0]` 합=9 → 각 라벨 목표 `floor(N * w/9)`
* 남는 개수는 **소수점 큰 순서**로 POS → HN → EN에 1개씩 배분
* 예) N=10 → POS 7, HN 3, EN 0 / N=5 → POS 3, HN 2, EN 0

---

## 6) 엑셀 쓰기(제출용)

각 질문을 한 행으로 확장, 원본 메타 복사
`약제분류번호=code, 약제 분류명=code_name, 구분=title, 세부인정기준 및 방법=슬라이스 텍스트`
`question` = 정제된 질문, `라벨` = `POSITIVE | HARD_NEGATIVE`

---

## 7) 실패·견고화

* 생성 줄 수 부족(10줄 미만) → **temperature ↑**, 프롬프트에 "produce more lines" 1줄 추가, **최대 2회 재시도**
* 429/timeout → **지수 백오프**(2s→4→8→최대 20s), **최대 3회**
* 문두 과다·대명사 위반 → 해당 라벨만 **부분 재생성**
* `brand_names` 없음(약제) → **주 약제명 중심**으로 POS, HN은 facet 위주

---

## 8) HN 변형기(코드 주도) — 요약 스펙

* **고정(유지)**: 성분/주 약제명, 가능하면 제형/경로, 품명/코드 토큰
* **변경(facet 중 1개만)**: 적응증, 대상/연령, 기간/횟수 경계, 용량/간격, 경구↔주사, 급여↔본인부담, 사전승인, 선행요법, 모니터링
* **출력**: 변형 문장을 **HN rewriter 프롬프트**로 보내 자연어 질문으로 다듬기
* **체커**: 고정 토큰 포함 확인 + facet 1개만 변경 확인(두 개 이상이면 실패)

---

## 9) 실행 순서(Claude Code용 체크리스트)

1. 엑셀 로드 → **표준 컬럼 매핑**
2. 본문 **슬라이싱(2~3k자)**
3. **POS 생성**(해당 프롬프트, 줄단위) → **후처리 게이트**
4. 상위 POS **3~5개를 앵커로** 선택 → **HN 변형기(코드)** → **HN rewriter** → **후처리 게이트**
5. **라벨 6:3:0 정규화**로 컷/보강
6. **엑셀 저장**(약제 스키마)

---

### 부록) 빠른 검증 규칙(정규식 요약)

* **대명사 금지**: `(이것|그것|해당|본|동)\s*(약|제제|제품)|\b(이것|그것)\b`
* **단위/정책어 탐지**: `\d|mg|㎎|U/L|%|회|개월|일|주|급여|비급여|본인부담|사전승인|수가|코드|기간|횟수`

---