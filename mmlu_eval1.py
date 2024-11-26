import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from categories import categories, subcategories

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
    
    # 시스템 메시지
    system_msg = "You are a helpful assistant that answers multiple choice questions. Respond with only a single letter (A, B, C, or D)."
    
    for idx, row in test_df.iterrows():
        # 프롬프트 생성
        k = args.ntrain
        prompt_end = format_example(test_df, idx, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        
        # verbose 모드일 때만 few-shot 예제와 테스트 질문 출력
        if args.verbose:
            print("\nFew-shot examples and test question:")
            print(prompt)
            print("=" * 80)
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        # 모델 응답 획득
        start_time = time.time()
        response = llm.invoke(messages)
        model_response = extract_answer(response.content)
        
        # 답변 평가
        answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        model_answer_num = answer_map.get(model_response, -1)
        
        is_correct = model_answer_num == int(row['answer'])
        correct_count += 1 if is_correct else 0
        total_count += 1
        accuracy = (correct_count / total_count) * 100
        cors.append(is_correct)
        
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

    return np.array(cors), accuracy

def main(args):
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
    subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
    cat_cors = {cat: [] for cat in categories}

    # 테스트할 과목 수 제한
    subjects = sorted(list(set(mmlu_dataset['test']['subject'])))
    if args.nsubjects > 0:  # nsubjects 파라미터 추가 필요
        subjects = subjects[:args.nsubjects]
    
    for subject in subjects:
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
        cors, acc = eval(args, subject, llm, dev_df, test_df)
        
        # 결과 저장
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

    # 최종 결과 계산 및 저장
    results = {"subcategories": {}, "categories": {}}
    
    for subcat in subcat_cors:
        if subcat_cors[subcat]:
            subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
            results["subcategories"][subcat] = subcat_acc
            print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        if cat_cors[cat]:
            cat_acc = np.mean(np.concatenate(cat_cors[cat]))
            results["categories"][cat] = cat_acc
            print("Average accuracy {:.3f} - {}".format(cat_acc, cat))

    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # 결과를 JSON 파일로 저장
    results_file = os.path.join(args.save_dir, f"accuracies_{args.model_name.replace('/', '_')}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--ntest", "-n", type=int, default=-1, 
                      help="Number of test samples per subject. -1 for all samples")
    parser.add_argument("--nsubjects", "-ns", type=int, default=-1,
                      help="Number of subjects to test. -1 for all subjects")  # 추가된 파라미터
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--base_url", type=str, default="http://localhost:1234/v1")
    parser.add_argument("--api_key", type=str, default="lm-studio")
    parser.add_argument("--model_name", type=str, default="llama-3-8b-instruct@q4_k_m")
    parser.add_argument("--verbose", "-v", action="store_true", 
                      help="출력 로그에 few-shot 예제를 포함합니다")
    args = parser.parse_args()
    main(args) 