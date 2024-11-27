MMLU Test 소스코드

HuggingFace cais/mmlu dataset을 통해서 테스트
LLM Model (API방식) 에 대해서 평가 수행

# 실행 예시

# arg 변수로 설정을 받는 모드
> python mmlu_eval1.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@iq2_s

# 사용자 입력을 직접 받는 모드
> python mmlu_eval2.py 