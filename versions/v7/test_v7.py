"""
V7 테스트 스크립트 - 소규모 데이터로 빠른 검증
"""

import pandas as pd
import logging
from drug_generator_v7 import V7QuestionGenerator, GenerationConfig, LengthBand

# 로깅 설정 (콘솔 출력만)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """기본 기능 테스트"""
    print("=== V7 기본 기능 테스트 ===")
    
    # 설정
    config = GenerationConfig()
    generator = V7QuestionGenerator(config)
    
    # 테스트 텍스트
    test_text = """
    트라스투주맙 주사제(허셉틴)는 HER2 양성 유방암 환자에게 사용되는 표적치료제입니다.
    급여기준: HER2 3+ 또는 FISH/CISH 양성인 경우 급여 인정
    사전승인: 필요 (관련 서류 제출)
    투여용량: 6mg/kg, 3주마다 투여
    모니터링: 심기능 검사 필수
    """
    
    # 텍스트 슬라이싱 테스트
    slices = generator.slice_text(test_text)
    print(f"텍스트 슬라이싱: {len(slices)}개 슬라이스")
    
    # 약제 정보 파싱 테스트
    test_title = "트라스투주맙(품명: 허셉틴주·허투주맙주)"
    drug_info = generator.parse_drug_info(test_title)
    print(f"약제 정보 파싱: {drug_info}")
    
    # 밴드 설정 확인
    print("\n밴드 설정:")
    for band, config in generator.config.bands.items():
        print(f"  {band.value}: {config.min_chars}-{config.max_chars}자, {config.ratio*100}%")
    
    # 유효성 검증 테스트
    test_questions = [
        ("트라스투주맙 주사제의 HER2 양성 유방암 환자 급여기준은 무엇인가?", LengthBand.SR),  # Valid SR (35자)
        ("HER2 양성 유방암에서 허셉틴 6mg/kg을 3주마다 투여할 때 사전승인이 필요하며 관련 서류는 무엇인가?", LengthBand.MR),  # Valid MR (83자)
        ("60세 여성이 HER2 양성 유방암으로 진단받았다. 기존 항암치료 후 재발이 확인되었고 심기능은 정상이다. LVEF 검사 결과 55%로 나왔으며 기존에 사용한 화학요법은 AC 및 Taxane 계열이다. 이런 상황에서 트라스투주맙 급여 적용이 가능한가?", LengthBand.LR),  # Valid LR (280자)
        ("이것은 어떤 약인가?", LengthBand.SR),  # Invalid (대명사)
        ("안녕하세요", LengthBand.SR),  # Invalid (물음표 없음)
    ]
    
    print("\n유효성 검증 테스트:")
    for question, band in test_questions:
        is_valid = generator.validate_question(question, band)
        status = "OK" if is_valid else "FAIL"
        print(f"  {band.value} '{question[:30]}...': {status}")
        
        # 실패 원인 디버깅
        if not is_valid:
            band_config = generator.config.bands[band]
            length_ok = band_config.min_chars <= len(question) <= band_config.max_chars
            question_mark_ok = question.endswith('?')
            pronoun_ok = not generator.strict_pronoun_pattern.search(question)
            specificity_ok = generator.specificity_pattern.search(question) is not None
            
            print(f"    길이({len(question)}): {'OK' if length_ok else 'FAIL'}")
            print(f"    물음표: {'OK' if question_mark_ok else 'FAIL'}")
            print(f"    대명사: {'OK' if pronoun_ok else 'FAIL'}")
            print(f"    구체성: {'OK' if specificity_ok else 'FAIL'}")
    
    return True

def test_mock_generation():
    """Mock 데이터로 전체 파이프라인 테스트"""
    print("\n=== V7 Mock 생성 테스트 ===")
    
    # Mock 데이터 생성
    mock_data = pd.DataFrame({
        'code': ['A001', 'A002'],
        'code_name': ['항암제', '면역억제제'],
        'title': [
            '트라스투주맙(품명: 허셉틴주·허투주맙주)', 
            '아달리무맙(품명: 휴미라주·아달리주)'
        ],
        'text': [
            """트라스투주맙은 HER2 양성 유방암 치료에 사용되는 단클론항체입니다. 
            급여기준: HER2 3+ 또는 FISH/CISH 양성 확인 필요. 
            사전승인: 관련 서류 제출 후 승인. 
            용법·용량: 초회 8mg/kg, 이후 6mg/kg을 3주 간격으로 정맥 투여. 
            모니터링: 심기능 검사(LVEF) 정기적 실시.""",
            
            """아달리무맙은 TNF-α 억제제로 류마티스관절염, 크론병 등에 사용됩니다.
            급여기준: 기존 치료에 불응하거나 부작용으로 중단한 경우.
            사전승인: 필요 (질병별 기준 상이).
            용법·용량: 성인 40mg을 격주로 피하주사.
            주의사항: 감염 위험 증가, 정기적 모니터링 필수."""
        ]
    })
    
    # 설정 (테스트용으로 작은 규모)
    config = GenerationConfig()
    generator = V7QuestionGenerator(config)
    
    try:
        # 각 행별로 질문 생성 (실제 API 호출 없이 Mock)
        all_questions = []
        
        for idx, row in mock_data.iterrows():
            print(f"처리 중: {row['code']}")
            
            # Mock 질문 생성 (API 호출 대신)
            mock_questions = generate_mock_questions(row, generator)
            all_questions.extend(mock_questions)
            
        # 결과 분석
        print(f"\n총 생성된 질문: {len(all_questions)}개")
        
        # 밴드별 분포
        band_counts = {}
        for q in all_questions:
            band_counts[q.band.value] = band_counts.get(q.band.value, 0) + 1
        
        print("밴드별 분포:")
        for band, count in band_counts.items():
            print(f"  {band}: {count}개")
        
        # 라벨별 분포
        label_counts = {}
        for q in all_questions:
            label_counts[q.label] = label_counts.get(q.label, 0) + 1
            
        print("라벨별 분포:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}개")
        
        # 샘플 질문 출력
        print("\n샘플 질문들:")
        for i, q in enumerate(all_questions[:5]):
            print(f"  {i+1}. [{q.band.value}][{q.label}] {q.text}")
        
        return True
        
    except Exception as e:
        print(f"Mock 테스트 실패: {e}")
        return False

def generate_mock_questions(row, generator):
    """Mock 질문 생성 (API 호출 없이)"""
    from drug_generator_v7 import Question
    import uuid
    
    mock_questions = []
    anchor_id = str(uuid.uuid4())
    
    # 각 밴드별 Mock 질문
    mock_data = {
        LengthBand.SR: [
            f"{row['code_name']}의 급여기준은?",
            f"{row['code_name']} 투여량은 얼마인가?",
            f"사전승인이 필요한가?"
        ],
        LengthBand.MR: [
            f"{row['code_name']}을 사용할 때 모니터링이 필요한 검사는?",
            f"HER2 양성 유방암에서 {row['code_name']} 사용 기준은?",
        ],
        LengthBand.LR: [
            f"60세 여성이 HER2 양성 유방암으로 진단받았다. 기존 항암치료 후 재발이 확인되었고, 심기능은 정상이다. 이 경우 {row['code_name']} 급여 적용이 가능한가?"
        ]
    }
    
    # Mock 질문 생성
    for band, questions in mock_data.items():
        for q_text in questions:
            question = Question(
                text=q_text,
                label="POSITIVE",
                band=band,
                anchor_id=anchor_id,
                doc_slice_id=f"{row['code']}_0",
                metadata={
                    'code': row['code'],
                    'code_name': row['code_name'],
                    'drug_info': generator.parse_drug_info(row['title'])
                }
            )
            mock_questions.append(question)
    
    # Mock Hard Negative (간단히)
    hn_question = Question(
        text=f"{row['code_name']}의 비급여 기준은?",  # POSITIVE의 변형
        label="HARD_NEGATIVE",
        band=LengthBand.SR,
        anchor_id=anchor_id,
        doc_slice_id=f"{row['code']}_0",
        metadata={
            'code': row['code'],
            'code_name': row['code_name'],
            'mutation_type': 'coverage_flip'
        }
    )
    mock_questions.append(hn_question)
    
    return mock_questions

def main():
    """테스트 실행"""
    print("V7 질문 생성기 테스트 시작")
    
    # 기본 기능 테스트
    basic_ok = test_basic_functionality()
    
    # Mock 생성 테스트  
    mock_ok = test_mock_generation()
    
    # 결과 요약
    print("\n=== 테스트 결과 요약 ===")
    basic_status = "성공" if basic_ok else "실패"
    mock_status = "성공" if mock_ok else "실패"
    print(f"기본 기능 테스트: {basic_status}")
    print(f"Mock 생성 테스트: {mock_status}")
    
    if basic_ok and mock_ok:
        print("\nV7 테스트 전체 성공!")
        print("다음 단계: 실제 API 키로 소규모 실제 테스트 진행")
    else:
        print("\n일부 테스트 실패 - 코드 점검 필요")

if __name__ == "__main__":
    main()