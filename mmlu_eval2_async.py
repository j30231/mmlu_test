# Async version (병렬 처리로 한꺼번에 10개씩 처리)

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from datasets import load_dataset
from openai import AsyncOpenAI
from categories import categories, subcategories
import random
import asyncio

# 기본 설정
choices = ["A", "B", "C", "D"]

# subject_to_category 딕셔너리 생성
subject_to_category = {}
for subject, subcats in subcategories.items():
    for cat, subcat_list in categories.items():
        if any(subcat in subcat_list for subcat in subcats):
            subject_to_category[subject] = cat
            break

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

async def async_evaluate_batch(client, messages_batch):
    """배치 단위로 비동기 평가를 수행합니다."""
    async def process_single_message(messages):
        try:
            response = await client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                temperature=0,
                max_tokens=10  # 응답 길이 제한
            )
            return response.choices[0].message
        except Exception as e:
            return e

    return await asyncio.gather(*[process_single_message(messages) for messages in messages_batch])

async def async_eval(args, subject, client, dev_df, test_df):
    correct_count = 0
    total_count = 0
    cors = []
    evaluation_results = []
    
    # 시스템 메시지 수정
    system_msg = """You are a helpful assistant that answers multiple choice questions. 
    Answer with ONLY a single letter (A, B, C, or D) without any explanation."""
    
    # 배치 크기 설정
    BATCH_SIZE = 10  # 한 번에 처리할 쿼리 수
    
    # 메시지 배치 준비
    messages_batch = []
    batch_indices = []
    
    for idx, row in test_df.iterrows():
        # few-shot 예제 준비
        if args.use_few_shot and args.ntrain > 0 and not dev_df.empty:
            k = args.ntrain
            prompt = f"Answer the following multiple choice questions about {format_subject(subject)}.\n\n"
            prompt += gen_prompt(dev_df, subject, k)
            prompt += "Now answer this question:\n\n"
            prompt += format_example(test_df, idx, include_answer=False)
        else:
            prompt = f"Answer this multiple choice question about {format_subject(subject)}:\n\n"
            prompt += format_example(test_df, idx, include_answer=False)
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        messages_batch.append(messages)
        batch_indices.append(idx)
        
        # 배치가 가득 찼거나 마지막 항목인 경우 처리
        if len(messages_batch) == BATCH_SIZE or idx == len(test_df) - 1:
            start_time = time.time()
            
            # 배치 평가 수행
            responses = await async_evaluate_batch(client, messages_batch)
            
            # 결과 처리
            for batch_idx, (response, test_idx) in enumerate(zip(responses, batch_indices)):
                try:
                    if isinstance(response, Exception):
                        print(f"Error processing question {test_idx}: {str(response)}")
                        continue
                        
                    model_response = response.content.strip()
                    model_answer = extract_answer(model_response)
                    
                    if model_answer is None:
                        print(f"Warning: 모델 응답에서 답변을 추출할 수 없습니다: {model_response}")
                        continue
                        
                    model_answer_num = ord(model_answer) - ord('A')
                    correct_answer = test_df.iloc[test_idx]['answer']
                    is_correct = model_answer_num == correct_answer
                    
                    if is_correct:
                        correct_count += 1
                    total_count += 1
                    
                    cors.append(is_correct)
                    accuracy = (correct_count / total_count) * 100
                    
                    # 결과 저장
                    result_entry = {
                        'subject': subject,
                        'question': test_df.iloc[test_idx]['question'],
                        'choices': [
                            test_df.iloc[test_idx]['choice_0'],
                            test_df.iloc[test_idx]['choice_1'],
                            test_df.iloc[test_idx]['choice_2'],
                            test_df.iloc[test_idx]['choice_3']
                        ],
                        'model_answer': model_response,
                        'correct_answer': correct_answer,
                        'is_correct': is_correct,
                        'response_time': time.time() - start_time,
                        'few_shot_used': args.use_few_shot and args.ntrain > 0,
                        'num_few_shot': args.ntrain if args.use_few_shot else 0
                    }
                    evaluation_results.append(result_entry)
                    
                    print(f"""
================================================================================
Progress: {total_count}/{len(test_df)} ({total_count/len(test_df)*100:.2f}%)
Subject: {subject}
Question: {test_df.iloc[test_idx]['question']}
Model answer: {model_response} ({model_answer_num})
Correct answer: {choices[int(correct_answer)]} ({correct_answer})
Result: {'Correct' if is_correct else 'Wrong'}
Accuracy so far: {accuracy:.2f}%
Time per batch: {(time.time() - start_time)/len(messages_batch):.2f}s
================================================================================
""")
                except Exception as e:
                    print(f"Error processing result for question {test_idx}: {str(e)}")
            
            # 배치 초기화
            messages_batch = []
            batch_indices = []
    
    # 결과 처리 부분 수정
    if total_count == 0:
        print(f"\n경고: {subject}에 대한 모든 질문 처리가 실패했습니다.")
        return np.array([]), 0.0, evaluation_results
    
    return np.array(cors), (correct_count / total_count) * 100, evaluation_results

def get_interactive_selections():
    """대화형으로 카테고리와 과목을 선택받는 함수"""
    # 카테고리 선택
    print("\n사용 가능한 카테고리:")
    category_list = list(categories.keys())
    for idx, category in enumerate(category_list, start=1):
        subject_count = len([s for s, subcats in subcategories.items() 
                           if any(subcat in categories[category] for subcat in subcats)])
        print(f"{idx}. {category} ({subject_count} subjects)")
    
    while True:
        try:
            cat_input = input("\n카테고리 번호를 선택하세요 (1-4, 여러 개는 쉼표로 구분, all): ").strip()
            if cat_input.lower() == 'all':
                selected_categories = category_list
                break
            
            indices = [int(i.strip()) for i in cat_input.split(',')]
            if all(1 <= i <= len(categories) for i in indices):
                selected_categories = [category_list[i-1] for i in indices]
                break
            print(f"1부터 {len(categories)} 사이의 숫자를 입력해주세요.")
        except ValueError:
            print("올바른 형식으로 입력해주세요 (예: 1,2 또는 all)")
    
    # 선택된 카테고리의 과목 표시
    selected_subjects = []
    for category in selected_categories:
        print(f"\n=== {category} 카테고리의 과목들: ===")
        category_subjects = [subject for subject, subcats in subcategories.items() 
                           if any(subcat in categories[category] for subcat in subcats)]
        
        for idx, subject in enumerate(category_subjects, 1):
            print(f"{idx}. {subject}")
        
        while True:
            try:
                subj_input = input(f"\n과목 번호를 선택하세요 (1-{len(category_subjects)}, 여러 개는 쉼표로 구분, all): ").strip()
                if subj_input.lower() == 'all':
                    selected_subjects.extend(category_subjects)
                    break
                
                indices = [int(i.strip()) for i in subj_input.split(',')]
                if all(1 <= i <= len(category_subjects) for i in indices):
                    selected_subjects.extend([category_subjects[i-1] for i in indices])
                    break
                print(f"1부터 {len(category_subjects)} 사이의 숫자를 입력해주세요.")
            except ValueError:
                print("올바른 형식으로 입력해주세요 (예: 1,2 또는 all)")
    
    # Few-shot 설정
    while True:
        use_few_shot = input("\nFew-shot 학습을 사용하시겠습니까? (y/n): ").lower().strip()
        if use_few_shot in ['y', 'n']:
            break
        print("y 또는 n으로 입력해주세요.")
    
    ntrain = 0
    if use_few_shot == 'y':
        while True:
            try:
                ntrain = int(input("Few-shot 예제 개수를 입력하세요 (1-5): "))
                if 1 <= ntrain <= 5:
                    break
                print("1부터 5 사이의 숫자를 입력해주세요.")
            except ValueError:
                print("올바른 숫자를 입력해주세요.")
    
    # 테스트 문제 개수 설정
    while True:
        try:
            ntest = int(input("\n각 과목당 평가할 문제 수를 입력하세요 (-1: 전체): "))
            if ntest == -1 or ntest > 0:
                break
            print("양수 또는 -1을 입력해주세요.")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
    
    return selected_subjects, use_few_shot == 'y', ntrain, ntest

def main(args):
    # 시작 시간 기록 및 타임스탬프 생성
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    
    # 결과 저장 디렉토리 구조 설정
    results_dir = os.path.join(args.save_dir, args.model_name.replace('/', '_'))
    interim_dir = os.path.join(results_dir, f"interim_{timestamp}")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(interim_dir):
        os.makedirs(interim_dir)

    # results 딕셔너리 초기화 추가
    results = {
        "subjects": {},
        "categories": {
            cat: {"subjects": [], "correct_rate": 0.0} 
            for cat in categories
        }
    }

    if args.auto_mode:
        # CLI 모드
        category_list = list(categories.keys())
        
        # 카테고리 처리
        if args.categories:
            try:
                cat_indices = [int(i.strip()) for i in args.categories.split(',')]
                selected_categories = [category_list[i-1] for i in cat_indices 
                                    if 1 <= i <= len(category_list)]
                # 선택된 카테고리에 속하는 과목만 필터링
                selected_subjects = [
                    subject for subject, cat in subject_to_category.items()
                    if cat in selected_categories
                ]
                
                # subjects 파라미터로 특정 번호의 과목 선택
                if args.subjects:
                    subject_indices = [int(i.strip()) for i in args.subjects.split(',')]
                    selected_subjects = [subject for i, subject in enumerate(selected_subjects, 1)
                                      if i in subject_indices]
                
                # Few-shot 설정 출력
                print(f"\n=== 평가 설정 ===")
                print(f"선택된 카테고리: {', '.join(selected_categories)}")
                print(f"선택된 과목 ({len(selected_subjects)}개): {', '.join(selected_subjects)}")
                print(f"Few-shot 사용: {args.use_few_shot}")
                print(f"테스트 문제 수: {'모두' if args.ntest <= 0 else args.ntest}")
                print("=" * 30 + "\n")
                
            except (ValueError, IndexError):
                print("카테고리 번호가 잘못되었습니다. 1-4 사이의 숫자를 사용하세요.")
                return
        else:
            selected_categories = category_list
            selected_subjects = list(subject_to_category.keys())
    else:
        # 대화형 모드
        selected_subjects, use_few_shot, ntrain, ntest = get_interactive_selections()
        args.use_few_shot = use_few_shot
        args.ntrain = ntrain
        args.ntest = ntest
        # 선택된 과목들의 카테고리 찾기
        selected_categories = list(set(subject_to_category[subject] for subject in selected_subjects))

    # 나머지 코드는 그대로 유지
    if args.nsubjects > 0:
        selected_subjects = random.sample(selected_subjects, min(args.nsubjects, len(selected_subjects)))

    # NumPy 타입 변환 함수 정의
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # OpenAI 클라이언트 초기화
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key
    )

    # MMLU 데이터셋 로드
    mmlu_dataset = load_dataset('cais/mmlu', 'all')
    
    # 결과 저장 디렉토리 생성
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 평가 결과 저장을 위한 변수들
    all_cors = []
    all_results = []
    subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
    cat_cors = {cat: [] for cat in categories}

    # 각 과목별 평가 수행
    for subject in selected_subjects:
        print(f"\n=== Evaluating {subject} ===\n")
        
        # 개발 데이터 준비 - few-shot 설정에 따라 조정
        if args.use_few_shot and args.ntrain > 0:
            dev_samples = [x for x in mmlu_dataset['dev'] if x['subject'] == subject][:args.ntrain]
        else:
            dev_samples = []  # few-shot을 사용지 않을 경 빈 리스트로 설정
            
        print(f"Number of few-shot examples: {len(dev_samples)}")  # 메시지 변경
        
        dev_data = {
            'question': [],
            'choice_0': [],
            'choice_1': [],
            'choice_2': [],
            'choice_3': [],
            'answer': []
        }
        
        # few-shot 예제가 있을 때만 dev_data를 채움
        if dev_samples:
            for sample in dev_samples:
                dev_data['question'].append(sample['question'])
                for i, choice in enumerate(sample['choices']):
                    dev_data[f'choice_{i}'].append(choice)
                dev_data['answer'].append(sample['answer'])
        
        # 테스트 데이터 준비
        test_samples = [x for x in mmlu_dataset['test'] if x['subject'] == subject]
        if args.ntest > 0:
            test_samples = test_samples[:args.ntest]
        print(f"Number of test examples: {len(test_samples)}")
        
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
        cors, acc, subject_results = asyncio.run(async_eval(args, subject, client, dev_df, test_df))
        
        # 결과가 비어있는 경우 처리
        if len(cors) == 0:
            print(f"경고: {subject} 평가 결과가 없습니다. 다음 과목으로 진행합니다.")
            continue
            
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
            "correct_rate": acc,
            "category": category,
            "details": subject_results
        }
        results["categories"][category]["subjects"].append({
            "subject": subject,
            "correct_rate": acc
        })

        #  서브젝트 완료 후 중간 결과 저장
        interim_results = {
            "metadata": {
                "model_name": args.model_name,
                "ntrain": args.ntrain,
                "ntest": args.ntest,
                "timestamp": time.strftime("%Y%m%d-%H%M%S")
            },
            "subject": subject,
            "category": category,
            "correct_rate": float(acc),  # NumPy float을 Python float으로 변환
            "evaluation_results": [
                {
                    "question": result["question"],
                    "model_answer": result["model_answer"],
                    "correct_answer": int(result["correct_answer"]),  # NumPy int를 Python int로 변환
                    "is_correct": bool(result["is_correct"]),  # NumPy bool을 Python bool로 변환
                    "response_time": float(result["response_time"])  # NumPy float을 Python float로 변환
                }
                for result in subject_results
            ]
        }
        
        # 중간 결과 파일 저장
        interim_save_path = os.path.join(
            interim_dir,
            f"interim_{subject}.json"
        )
        
        with open(interim_save_path, 'w', encoding='utf-8') as f:
            json.dump(interim_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n중간 결과가 저장되었습니다: {interim_save_path}")

    # 카테고리별 평균 정답률 계산
    for cat in categories:
        cat_subjects = results["categories"][cat]["subjects"]
        if cat_subjects:
            cat_rate = np.mean([s["correct_rate"] for s in cat_subjects])
            results["categories"][cat]["correct_rate"] = round(cat_rate, 2)
            print(f"\n평균 정답률 {cat_rate:.2f} - {cat}")

    # 전체 평균 정답률 계산
    all_rates = [s["correct_rate"] for s in results["subjects"].values()]
    if all_rates:
        results["overall_correct_rate"] = round(np.mean(all_rates), 2)
        print(f"\n전체 정답률: {results['overall_correct_rate']:.2f}")

    # 최종 결과 저장 전에 total_time 계산
    total_time = time.time() - start_time
    
    # 최종 결과 저장 전에 데이터 구조 정리
    final_results = {
        "metadata": {
            "model_name": args.model_name,
            "ntrain": args.ntrain,
            "ntest": args.ntest,
            "timestamp": time.strftime("%Y%m%d-%H%M%S")
        },
        "categories": {},
        "overall_correct_rate": float(round(results["overall_correct_rate"], 2))
    }

    # 카테고리별 결과 정리
    for cat in categories:
        cat_subjects = results["categories"][cat]["subjects"]
        if cat_subjects:
            final_results["categories"][cat] = {
                "correct_rate": float(round(results["categories"][cat]["correct_rate"], 2)),
                "subjects": [
                    {
                        "subject": s["subject"],
                        "correct_rate": float(round(s["correct_rate"], 2))
                    }
                    for s in cat_subjects
                ]
            }

    # 최종 결과 JSON 파일 저장
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    save_path = os.path.join(
        results_dir,
        f"results_{timestamp}.json"
    )
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # 평균 정답률 요약 파일 저장
    summary = {
        "timestamp": timestamp,
        "model_name": args.model_name,
        "total_time": f"{total_time:.1f}s",
        "overall_correct_rate": float(round(results["overall_correct_rate"], 2)),
        "category_correct_rates": {
            cat: float(round(results["categories"][cat]["correct_rate"], 2))
            for cat in categories if results["categories"][cat]["subjects"]
        },
        "few_shot_used": args.use_few_shot,
        "num_few_shot": args.ntrain if args.use_few_shot else 0,
        "num_test": args.ntest,
        "selected_categories": selected_categories,
        "num_subjects": len(selected_subjects)
    }
    
    summary_path = os.path.join(args.save_dir, "correct_summary.jsonl")
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(summary, ensure_ascii=False) + '\n')
    
    print(f"\n결과가 저장되었습니다: {save_path}")
    print(f"정답률 요약이 추가되었습니다: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto_mode", "-am", action="store_true",
                      help="CLI 인자를 사용하여 자동으로 설정합니다")
    parser.add_argument("--categories", "-c", type=str,
                      help="평가할 카테고리 (쉼표로 구분)")
    parser.add_argument("--subjects", "-sb", type=str,
                      help="평가할 과목 (쉼표로 구분)")
    parser.add_argument("--use_few_shot", "-f", action="store_true", default=False,
                      help="few-shot 학습 사용 여부")
    parser.add_argument("--ntrain", "-k", type=int, default=0,
                      help="few-shot 예제 개수")
    parser.add_argument("--ntest", "-n", type=int, default=-1,
                      help="각 과목당 평가할 문제 수")
    parser.add_argument("--nsubjects", "-ns", type=int, default=-1)
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--base_url", type=str, default="http://localhost:1234/v1")
    parser.add_argument("--api_key", type=str, default="lm-studio")
    parser.add_argument("--model_name", type=str, default="llama-3-8b-instruct@?")
    parser.add_argument("--verbose", "-v", action="store_true", 
                      help="출력 로그에 few-shot 예제를 포함합니다")
    args = parser.parse_args()
    main(args)
