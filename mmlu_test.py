from datasets import load_dataset
from langchain_openai import ChatOpenAI
import random
import json
from mmul_category import MMLU_CATEGORIES, SUBJECT_TO_CATEGORY

mmlu_dataset = load_dataset('cais/mmlu', 'all')

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="llama-3-8b-instruct@q4_k_m",
    streaming=True,
)

def extract_answer(response):
    """
    모델 응답에서 A, B, C, D 형태의 답변을 추출합니다.
    """
    response = response.upper()
    
    # 답변 패턴을 찾습니다
    patterns = [
        r"THE CORRECT ANSWER IS ([A-D])",
        r"ANSWER: ([A-D])",
        r"([A-D])\. \d+",
        r"OPTION ([A-D])",
        r"CHOOSE ([A-D])",
        r"^([A-D])$"
    ]
    
    import re
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    
    # 마지막 수단: A, B, C, D가 언급된 마지막 위치를 찾습니다
    last_answer = None
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response:
            pos = response.rindex(letter)
            if last_answer is None or pos > last_answer[1]:
                last_answer = (letter, pos)
    
    return last_answer[0] if last_answer else None

def evaluate_mmlu(dataset, llm, test_subjects=None, ntrain=5):
    """
    MMLU 데이터셋에 대한 평가를 수행합니다.
    
    Args:
        dataset: MMLU 데이터셋
        llm: 평가할 언어 모델
        test_subjects: 평가할 과목 리스트 (None이면 전체 과목)
        ntrain: few-shot 학습에 사용할 예제 수 (기본값: 5)
    """
    correct = 0
    total = 0
    evaluation_results = []

    for category in selected_categories:
        count = sum(
            1 for sample in dataset['test']
            if SUBJECT_TO_CATEGORY.get(sample.get('subject'), 'Other') == category
        )
        print(f"카테고리 '{category}'의 샘플 수: {count}")

        category_samples = [
            sample for sample in dataset['test']
            if SUBJECT_TO_CATEGORY.get(sample.get('subject'), 'Other') == category
        ]
        if not category_samples:
            print(f"카테고리 '{category}'에 해당하는 샘플이 없습니다.")
            continue

        samples = random.sample(category_samples, min(num_samples_per_category, len(category_samples)))

        for sample in samples:
            question = sample['question']
            choices = sample['choices']
            correct_answer = sample['answer']
            subject = sample.get('subject')

            # 매핑 확인을 위한 출력
            mapped_category = SUBJECT_TO_CATEGORY.get(subject, 'Other')
            print(f"Sample Subject: {subject}, Mapped Category: {mapped_category}")

            # `answer` 필드의 타입 확인 및 변환
            if isinstance(correct_answer, int):
                correct_answer = chr(65 + correct_answer)  # 0 → 'A', 1 → 'B', 등
            elif isinstance(correct_answer, str):
                correct_answer = correct_answer.strip().upper()

            prompt = f"Question: {question}\n\nChoices:\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "\nPlease select the correct answer from A, B, C, D."

            response = llm.invoke(prompt)
            model_response = response.content.strip()
            
            # 모델 응답에서 답변 추출
            model_answer = extract_answer(model_response)
            if model_answer is None:
                print(f"Warning: 모델 응답에서 답변을 추출할 수 없습니다: {model_response}")
                continue

            print(f"질문: {question}")
            print(f"모델 응답: {model_response}")
            print(f"추출된 답변: {model_answer}, 정답: {correct_answer}")

            is_correct = (model_answer == correct_answer)
            if is_correct:
                correct += 1
            total += 1

            evaluation_results.append({
                'category': category,
                'question': question,
                'choices': choices,
                'correct_answer': correct_answer,
                'model_answer': model_answer,
                'model_response': model_response,
                'is_correct': is_correct
            })

    accuracy = correct / total if total > 0 else 0
    print(f"MMLU Accuracy: {accuracy:.2%}")

    # 평가 결과를 별도 파일에 저장
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    print("Available Categories:")
    for idx, category in enumerate(MMLU_CATEGORIES, start=1):
        print(f"{idx}. {category}")

    selected_indices = input("평가할 카테고리의 번호를 쉼표로 구분하여 입력하세요 (예: 1,3): ")
    selected_indices = selected_indices.replace(' ', '').split(',')
    selected_categories = [
        MMLU_CATEGORIES[int(idx)-1] for idx in selected_indices
        if idx.isdigit() and 1 <= int(idx) <= len(MMLU_CATEGORIES)
    ]

    if not selected_categories:
        print("유효한 카테고리가 선택되지 않았습니다.")
        exit(1)

    try:
        ntrain = int(input("few-shot 학습에 사용할 예제 수를 입력하세요 (기본값: 5): ") or "5")
        num_samples_per_category = int(input("각 카테고리당 평가할 샘플 수를 입력하세요: "))
    except ValueError:
        print("유효한 숫자를 입력하세요.")
        exit(1)

    # 데이터셋 샘플 확인
    print("테스트 샘플 예시:")
    print(mmlu_dataset['test'][0])

    # 모든 카테고리의 샘플 수 출력
    print("\n전체 카테고리의 샘플 수:")
    for category in MMLU_CATEGORIES:
        count = sum(
            1 for sample in mmlu_dataset['test']
            if SUBJECT_TO_CATEGORY.get(sample.get('subject'), 'Other') == category
        )
        print(f"카테고리 '{category}'의 전체 샘플 수: {count}")

    evaluate_mmlu(mmlu_dataset, llm, selected_categories, ntrain=ntrain)
