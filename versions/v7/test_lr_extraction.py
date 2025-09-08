"""
LR 추출 로직 테스트
"""

def test_lr_extraction():
    # 테스트 샘플
    test_content = """65세 남성이 류마티스관절염으로 진단받았다. 메토트렉세이트와 설파살라진 치료에도 불구하고 증상이 지속되고 있다. 관절 손상이 진행되고 있으며, TNF-α 억제제 사용을 고려하고 있다. 이 경우 아달리무맙 급여 기준을 충족하는가?

50세 여성이 크론병으로 진단받았다. 스테로이드 치료에 의존적이며 면역억제제에도 불응성을 보인다. 장기간 스테로이드 사용으로 인한 부작용이 우려되는 상황이다. 이러한 환자에게 아달리무맙 사용이 급여로 인정되는가?"""

    # LR 추출 로직 재현
    questions = []
    cases = test_content.strip().split('\n\n')
    
    print(f"총 {len(cases)}개 케이스 발견")
    
    for i, case in enumerate(cases):
        case = case.strip()
        if not case:
            continue
            
        print(f"\n=== 케이스 {i+1} ===")
        print(f"길이: {len(case)}자")
        print(f"내용: {case[:100]}...")
        
        # 전체 시나리오 길이 확인 (200-600자)
        if not (200 <= len(case) <= 600):
            print("❌ 길이 조건 불만족")
            continue
        else:
            print("✅ 길이 조건 만족")
            
        # ? 기호로 끝나는 문장 찾기
        lines = case.split('\n')
        print(f"라인 수: {len(lines)}")
        
        for j, line in enumerate(reversed(lines)):
            line = line.strip()
            print(f"  라인 {j}: '{line}' (끝: {line[-1] if line else 'None'})")
            if line.endswith('?'):
                questions.append(case)  # 전체 시나리오 반환
                print("✅ 질문 발견!")
                break
        else:
            print("❌ 질문을 찾을 수 없음")
            
    print(f"\n최종 결과: {len(questions)}개 질문 추출")
    
if __name__ == "__main__":
    test_lr_extraction()