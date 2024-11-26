import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from categories import categories, subcategories
import random

# 기본 설정
choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        # 숫자를 알파벳으로 변환
        answer_num = int(df.iloc[idx, k + 1])
        answer_letter = choices[answer_num]  # 0->A, 1->B, 2->C, 3->D
        prompt += " {}\n\n".format(answer_letter)
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def extract_answer(response_text: str) -> str:
    """모델 응답에서 A, B, C, D 형식의 답변을 추출합니다."""
    if not response_text:
        raise ValueError("모델 응답이 비어있습니다")
    
    # 응답을 줄 단위로 분리
    lines = response_text.upper().split('\n')
    
    # "ANSWER:" 또는 "THE ANSWER IS" 패턴 찾기
    for line in lines:
        if "ANSWER:" in line or "THE ANSWER IS" in line:
            # 해당 라인에서 A, B, C, D 찾기
            for char in line:
                if char in ['A', 'B', 'C', 'D']:
                    return char
    
    # 위 방법으로 찾지 못한 경우, 전체 텍스트에서 첫 번째로 나오는 A, B, C, D 찾기
    for char in response_text.upper():
        if char in ['A', 'B', 'C', 'D']:
            return char
            
    raise ValueError(f"유효한 답변(A,B,C,D)을 찾을 수 없습니다: {response_text[:100]}...")

def eval(args, subject, llm, dev_df, test_df):
    correct_count = 0
    total_count = 0
    cors = []
    evaluation_results = []
    
    # 시스템 메시지 수정
    system_msg = """You are a helpful assistant that answers multiple choice questions. 
    Answer with ONLY a single letter (A, B, C, or D) without any explanation."""
    
    for idx, row in test_df.iterrows():
        start_time = time.time()
        
        # few-shot 예제 준비 부분 수정
        if args.use_few_shot and args.ntrain > 0:
            k = args.ntrain
            # 테스트 문제를 마지막에 추가
            prompt = f"Answer the following multiple choice questions about {format_subject(subject)}.\n\n"
            prompt += gen_prompt(dev_df, subject, k)  # few-shot 예제들
            prompt += "Now answer this question:\n\n"
            prompt += format_example(test_df, idx, include_answer=False)
        else:
            prompt = f"Answer this multiple choice question about {format_subject(subject)}:\n\n"
            prompt += format_example(test_df, idx, include_answer=False)
        
        # verbose 모드일 때만 few-shot 예제와 테스트 질문 출력
        if args.verbose:
            print("\nFew-shot examples and test question:")
            print(prompt)
            print("=" * 80)
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        # 모델 응답 받기
        response = llm.invoke(messages)
        model_response = response.content.strip()
        
        # 답변 처리
        model_answer = extract_answer(model_response)
        if model_answer is None:
            print(f"Warning: 모델 응답에서 답변을 추출할 수 없습니다: {model_response}")
            continue
            
        model_answer_num = ord(model_answer) - ord('A')
        correct_answer = row['answer']
        is_correct = model_answer_num == correct_answer
        
        if is_correct:
            correct_count += 1
        total_count += 1
        
        cors.append(is_correct)
        accuracy = (correct_count / total_count) * 100

        # 결과 저장
        result_entry = {
            'subject': subject,
            'question': row['question'],
            'choices': [
                row['choice_0'],
                row['choice_1'],
                row['choice_2'],
                row['choice_3']
            ],
            'model_answer': model_response,
            'correct_answer': row['answer'],
            'is_correct': is_correct,
            'response_time': time.time() - start_time,
            'few_shot_used': args.use_few_shot and args.ntrain > 0,
            'num_few_shot': args.ntrain if args.use_few_shot else 0
        }
        evaluation_results.append(result_entry)
        
        print(f"""
================================================================================
Progress: {idx+1}/{len(test_df)} ({(idx+1)/len(test_df)*100:.1f}%)
Subject: {subject}
Question: {row['question']}
Model answer: {model_response} ({model_answer_num})
Correct answer: {choices[int(row['answer'])]} ({row['answer']})
Result: {'Correct' if is_correct else 'Wrong'}
Accuracy so far: {accuracy:.1f}%
Time: {time.time() - start_time:.1f}s
================================================================================
""")

    return np.array(cors), accuracy, evaluation_results

def main(args):
    # 카테고리 매핑 생성
    subject_to_category = {}
    for subject, subcats in subcategories.items():
        for cat, subcat_list in categories.items():
            if any(subcat in subcat_list for subcat in subcats):
                subject_to_category[subject] = cat
                break
    
    # selected_subjects 리스트를 미리 초기화
    selected_subjects = []
    
    if not args.auto_categories:
        print("\n사용 가능한 카테고리:")
        category_list = list(categories.keys())
        for idx, category in enumerate(category_list, start=1):
            subject_count = len([s for s, c in subject_to_category.items() if c == category])
            print(f"{idx}. {category} ({subject_count} subjects)")
        
        while True:
            try:
                selected_indices = input("\n평가할 카테고리의 번호를 쉼표로 구분하여 입력하세요 (예: 1,3 또는 all): ").strip()
                if selected_indices.lower() == 'all':
                    selected_categories = category_list
                    break
                
                indices = [int(idx.strip()) for idx in selected_indices.split(',')]
                selected_categories = [category_list[i-1] for i in indices if 1 <= i <= len(category_list)]
                
                if selected_categories:
                    break
                print("올바른 카테고리 번호를 입력해주세요.")
            except ValueError:
                print("올바른 형식으로 입력해주세요 (예: 1,3 또는 all)")
        
        # 선택된 카테고리의 과목 목록 표시 및 선택
        for category in selected_categories:
            category_subjects = [subject for subject, cat in subject_to_category.items() if cat == category]
            print(f"\n{category} 카테고리의 과목들:")
            for idx, subject in enumerate(category_subjects, start=1):
                print(f"{idx}. {subject}")
            
            while True:
                try:
                    subject_indices = input(f"\n평가할 과목의 번호를 쉼표로 구분하여 입력하세요 (예: 1,3 또는 all): ").strip()
                    if subject_indices.lower() == 'all':
                        selected_subjects.extend(category_subjects)
                        break
                    
                    indices = [int(idx.strip()) for idx in subject_indices.split(',')]
                    selected_subjects.extend([category_subjects[i-1] for i in indices if 1 <= i <= len(category_subjects)])
                    break
                except ValueError:
                    print("올바른 형식으로 입력해주세요 (예: 1,3 또는 all)")

    # few-shot 사용 여부 입력 받기
    while True:
        few_shot = input("\nfew-shot 학습을 사용하시겠습니까? (y/n): ").strip().lower()
        if few_shot in ['y', 'n']:
            args.use_few_shot = (few_shot == 'y')
            break
        print("'y' 또는 'n'으로 입력해주세요.")
    
    # few-shot 개수 입력 받기
    if args.use_few_shot:
        while True:
            try:
                ntrain = input("\nfew-shot 예제 개수를 입력하세요 (기본값: 5): ").strip()
                if not ntrain:  # 입력이 없으면 기본값 사용
                    args.ntrain = 5
                    break
                args.ntrain = int(ntrain)
                if args.ntrain >= 0:
                    break
                print("0 이상의 숫자를 입력해주세요.")
            except ValueError:
                print("올바른 숫자를 입력해주세요.")
    
    # 문제 수 입력 받기
    while True:
        try:
            num_questions = input("\n각 과목당 평가할 문제 수를 입력하세요 (기본값: 전체): ").strip()
            if not num_questions:  # 입력이 없으면 전체 문제 사용
                args.ntest = -1
                break
            args.ntest = int(num_questions)
            if args.ntest > 0:
                break
            print("1 이상의 숫자를 입력해주세요.")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")

    if args.nsubjects > 0:
        selected_subjects = random.sample(selected_subjects, min(args.nsubjects, len(selected_subjects)))

    # 결과 저장을 위한 딕셔너리 초기화
    results = {
        "metadata": {
            "model_name": args.model_name,
            "ntrain": args.ntrain,
            "ntest": args.ntest,
            "timestamp": time.strftime("%Y%m%d-%H%M%S")
        },
        "categories": {cat: {"accuracy": 0.0, "subjects": []} for cat in categories},
        "subjects": {},
        "overall_accuracy": 0.0
    }

    # LLM 모델 초기화
    llm = ChatOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model_name,
        streaming=True,
    )

    # MMLU 데이터셋 로드
    mmlu_dataset = load_dataset('cais/mmlu', 'all')
    
    # 결과 저장 디렉토리 생성
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    results_dir = os.path.join(args.save_dir, f"results_{args.model_name}")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 평가 결과 저장을 위한 변수들
    all_cors = []
    all_results = []
    subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
    cat_cors = {cat: [] for cat in categories}

    # 각 과목별 평가 수행
    for subject in selected_subjects:
        print(f"\n=== Evaluating {subject} ===\n")
        
        # 개발 데이터 준비
        dev_samples = [x for x in mmlu_dataset['dev'] if x['subject'] == subject][:args.ntrain]
        print(f"Number of training examples: {len(dev_samples)}")  # 훈련 예제 수 출력
        
        dev_data = {
            'question': [],
            'choice_0': [],
            'choice_1': [],
            'choice_2': [],
            'choice_3': [],
            'answer': []
        }
        
        for sample in dev_samples:
            dev_data['question'].append(sample['question'])
            for i, choice in enumerate(sample['choices']):
                dev_data[f'choice_{i}'].append(choice)
            dev_data['answer'].append(sample['answer'])
            
        # 테스트 데이터 준비
        test_samples = [x for x in mmlu_dataset['test'] if x['subject'] == subject]
        if args.ntest > 0:
            test_samples = test_samples[:args.ntest]
        print(f"Number of test examples: {len(test_samples)}")  # 테스트 예제 수 출력
        
        # test_data 딕셔너리 생성 추가
        test_data = {
            'question': [],
            'choice_0': [],
            'choice_1': [],
            'choice_2': [],
            'choice_3': [],
            'answer': []
        }
        
        for sample in test_samples:
            test_data['question'].append(sample['question'])
            for i, choice in enumerate(sample['choices']):
                test_data[f'choice_{i}'].append(choice)
            test_data['answer'].append(sample['answer'])
        
        # DataFrame 생성
        dev_df = pd.DataFrame(dev_data)
        test_df = pd.DataFrame(test_data)

        # 평가 수행
        cors, acc, subject_results = eval(args, subject, llm, dev_df, test_df)
        all_results.extend(subject_results)
        
        # 결과 저장
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        # 결과 저장
        category = subject_to_category[subject]
        results["subjects"][subject] = {
            "accuracy": acc,
            "category": category,
            "details": subject_results
        }
        results["categories"][category]["subjects"].append({
            "subject": subject,
            "accuracy": acc
        })

    # 카테고리별 평균 정확도 계산
    for cat in categories:
        cat_subjects = results["categories"][cat]["subjects"]
        if cat_subjects:
            cat_acc = np.mean([s["accuracy"] for s in cat_subjects])
            results["categories"][cat]["accuracy"] = cat_acc
            print(f"\nAverage accuracy {cat_acc:.3f} - {cat}")

    # 전체 평균 정확도 계산
    all_accuracies = [s["accuracy"] for s in results["subjects"].values()]
    if all_accuracies:
        results["overall_accuracy"] = np.mean(all_accuracies)
        print(f"\nOverall accuracy: {results['overall_accuracy']:.3f}")

    # 결과 저장
    save_path = os.path.join(
        args.save_dir,
        f"results_{args.model_name.replace('/', '_')}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과가 저장되었습니다: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5,
                      help="few-shot 예제 개수")
    parser.add_argument("--ntest", "-n", type=int, default=-1)
    parser.add_argument("--nsubjects", "-ns", type=int, default=-1)
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--base_url", type=str, default="http://localhost:1234/v1")
    parser.add_argument("--api_key", type=str, default="lm-studio")
    parser.add_argument("--model_name", type=str, default="llama-3-8b-instruct@?")
    parser.add_argument("--verbose", "-v", action="store_true", 
                      help="출력 로그에 few-shot 예제를 포함합니다")
    parser.add_argument("--auto_categories", "-a", action="store_true",
                      help="카테고리 선택을 자동으로 처리합니다")
    parser.add_argument("--use_few_shot", type=bool, default=True, 
                      help="few-shot 학습 사용 여부")
    args = parser.parse_args()
    main(args) 