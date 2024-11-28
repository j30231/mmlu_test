# Sync version

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

# Basic settings
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
        # Convert number to letter
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
    """Extracts A, B, C, D format answer from model response."""
    if not response_text:
        raise ValueError("Model response is empty")
    
    # Split response by lines
    lines = response_text.upper().split('\n')
    
    # Look for "ANSWER:" or "THE ANSWER IS" pattern
    for line in lines:
        if "ANSWER:" in line or "THE ANSWER IS" in line:
            # Find A, B, C, D in the line
            for char in line:
                if char in ['A', 'B', 'C', 'D']:
                    return char
    
    # If not found above, look for first A, B, C, D in entire text
    for char in response_text.upper():
        if char in ['A', 'B', 'C', 'D']:
            return char
            
    raise ValueError(f"Cannot find valid answer (A,B,C,D) in response: {response_text[:100]}...")

def eval(args, subject, llm, dev_df, test_df):
    correct_count = 0
    total_count = 0
    cors = []
    evaluation_results = []
    
    # Modify system message
    system_msg = """You are a helpful assistant that answers multiple choice questions. 
    Answer with ONLY a single letter (A, B, C, or D) without any explanation."""
    
    for idx, row in test_df.iterrows():
        start_time = time.time()
        
        # Prepare few-shot examples
        if args.use_few_shot and args.ntrain > 0 and not dev_df.empty:  # Apply few-shot only if dev_df is not empty
            k = args.ntrain
            # Add test question at the end
            prompt = f"Answer the following multiple choice questions about {format_subject(subject)}.\n\n"
            prompt += gen_prompt(dev_df, subject, k)  # few-shot examples
            prompt += "Now answer this question:\n\n"
            prompt += format_example(test_df, idx, include_answer=False)
        else:
            prompt = f"Answer this multiple choice question about {format_subject(subject)}:\n\n"
            prompt += format_example(test_df, idx, include_answer=False)
        
        # Print few-shot examples and test question in verbose mode only
        if args.verbose:
            print("\nFew-shot examples and test question:")
            print(prompt)
            print("=" * 80)
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        # Get model response
        response = llm.invoke(messages)
        model_response = response.content.strip()
        
        # Process answer
        model_answer = extract_answer(model_response)
        if model_answer is None:
            print(f"Warning: Could not extract answer from model response: {model_response}")
            continue
            
        model_answer_num = ord(model_answer) - ord('A')
        correct_answer = row['answer']
        is_correct = model_answer_num == correct_answer
        
        if is_correct:
            correct_count += 1
        total_count += 1
        
        cors.append(is_correct)
        accuracy = (correct_count / total_count) * 100

        # Save result
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

def get_interactive_selections():
    """Interactive function to select categories and subjects"""
    # Category selection
    print("\nAvailable categories:")
    category_list = list(categories.keys())
    for idx, category in enumerate(category_list, start=1):
        subject_count = len([s for s, subcats in subcategories.items() 
                           if any(subcat in categories[category] for subcat in subcats)])
        print(f"{idx}. {category} ({subject_count} subjects)")
    
    while True:
        try:
            cat_input = input("\nSelect category numbers (1-4, comma-separated, or 'all'): ").strip()
            if cat_input.lower() == 'all':
                selected_categories = category_list
                break
            
            indices = [int(i.strip()) for i in cat_input.split(',')]
            if all(1 <= i <= len(categories) for i in indices):
                selected_categories = [category_list[i-1] for i in indices]
                break
            print(f"Please enter numbers between 1 and {len(categories)}.")
        except ValueError:
            print("Please use correct format (e.g., 1,2 or 'all')")

    # Subject selection for each category
    selected_subjects = []
    for category in selected_categories:
        print(f"\n=== Subjects in {category} category: ===")
        category_subjects = [subject for subject, subcats in subcategories.items() 
                           if any(subcat in categories[category] for subcat in subcats)]
        
        for idx, subject in enumerate(category_subjects, 1):
            print(f"{idx}. {subject}")
        
        while True:
            try:
                subj_input = input(f"\nSelect subject numbers (1-{len(category_subjects)}, comma-separated, or 'all'): ").strip()
                if subj_input.lower() == 'all':
                    selected_subjects.extend(category_subjects)
                    break
                
                indices = [int(i.strip()) for i in subj_input.split(',')]
                if all(1 <= i <= len(category_subjects) for i in indices):
                    selected_subjects.extend([category_subjects[i-1] for i in indices])
                    break
                print(f"Please enter numbers between 1 and {len(category_subjects)}.")
            except ValueError:
                print("Please use correct format (e.g., 1,2 or 'all')")
    
    # Few-shot settings
    while True:
        use_few_shot = input("\nUse few-shot learning? (y/n): ").lower().strip()
        if use_few_shot in ['y', 'n']:
            break
        print("Please enter y or n.")
    
    ntrain = 0
    if use_few_shot == 'y':
        while True:
            try:
                ntrain = int(input("Enter number of few-shot examples (1-5): "))
                if 1 <= ntrain <= 5:
                    break
                print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")
    
    # Number of test questions
    while True:
        try:
            ntest = int(input("\nEnter number of test questions per subject (-1: all): "))
            if ntest == -1 or ntest > 0:
                break
            print("Please enter a positive number or -1.")
        except ValueError:
            print("Please enter a valid number.")
    
    return selected_subjects, use_few_shot == 'y', ntrain, ntest

def main(args):
    # Create category mapping
    subject_to_category = {}
    for subject, subcats in subcategories.items():
        for cat, subcat_list in categories.items():
            if any(subcat in subcat_list for subcat in subcats):
                subject_to_category[subject] = cat
                break
    
    if args.auto_mode:
        # CLI mode
        category_list = list(categories.keys())
        selected_subjects = []
        
        # Create full subject list (including category information)
        all_subjects = []
        for subject, cat in subject_to_category.items():
            all_subjects.append((cat, subject))
        
        # Category handling
        if args.categories:
            try:
                cat_indices = [int(i.strip()) for i in args.categories.split(',')]
                selected_categories = [category_list[i-1] for i in cat_indices 
                                    if 1 <= i <= len(category_list)]
            except (ValueError, IndexError):
                print("Invalid category number. Please use numbers between 1 and 4.")
                return
        else:
            selected_categories = category_list

        # Subject selection
        if args.subjects:
            try:
                subj_indices = [int(i.strip()) for i in args.subjects.split(',')]
                selected_subjects = [all_subjects[i-1][1] for i in subj_indices 
                                  if 1 <= i <= len(all_subjects)]
                if not selected_subjects:
                    print("No valid subjects selected.")
                    return
            except (ValueError, IndexError):
                print("Invalid subject number.")
                return
        else:
            selected_subjects = [subject for _, subject in all_subjects]

        # Print few-shot settings
        print(f"\n=== Evaluation Settings ===")
        print(f"Selected categories: {', '.join(selected_categories)}")
        print(f"Selected subjects ({len(selected_subjects)}): {', '.join(selected_subjects)}")
        print(f"Use few-shot: {args.use_few_shot}")
        print(f"Test questions: {'all' if args.ntest <= 0 else args.ntest}")
        print("=" * 30 + "\n")
    else:
        # Interactive mode
        selected_subjects, use_few_shot, ntrain, ntest = get_interactive_selections()
        args.use_few_shot = use_few_shot
        args.ntrain = ntrain
        args.ntest = ntest

    # Rest of the code
    if args.nsubjects > 0:
        selected_subjects = random.sample(selected_subjects, min(args.nsubjects, len(selected_subjects)))

    # Initialize dictionary for saving results
    results = {
        "metadata": {
            "model_name": args.model_name,
            "ntrain": args.ntrain,
            "ntest": args.ntest,
            "timestamp": time.strftime("%Y%m%d-%H%M%S")
        },
        "categories": {cat: {"correct_rate": 0.0, "subjects": []} for cat in categories},
        "subjects": {},
        "overall_correct_rate": 0.0
    }

    # Initialize LLM model
    llm = ChatOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model_name,
        streaming=True,
    )

    # Load MMLU dataset
    mmlu_dataset = load_dataset('cais/mmlu', 'all')
    
    # Create results directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Start time recording
    start_time = time.time()
    
    # Variables for saving evaluation results
    all_cors = []
    all_results = []
    subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
    cat_cors = {cat: [] for cat in categories}

    # Evaluate each subject
    for subject in selected_subjects:
        print(f"\n=== Evaluating {subject} ===\n")
        
        # Prepare development data - adjust based on few-shot settings
        if args.use_few_shot and args.ntrain > 0:
            dev_samples = [x for x in mmlu_dataset['dev'] if x['subject'] == subject][:args.ntrain]
        else:
            dev_samples = []  # No few-shot if not used
            
        print(f"Number of few-shot examples: {len(dev_samples)}")  # Change message
        
        dev_data = {
            'question': [],
            'choice_0': [],
            'choice_1': [],
            'choice_2': [],
            'choice_3': [],
            'answer': []
        }
        
        # Fill dev_data only if there are few-shot examples
        if dev_samples:
            for sample in dev_samples:
                dev_data['question'].append(sample['question'])
                for i, choice in enumerate(sample['choices']):
                    dev_data[f'choice_{i}'].append(choice)
                dev_data['answer'].append(sample['answer'])
        
        # Prepare test data
        test_samples = [x for x in mmlu_dataset['test'] if x['subject'] == subject]
        if args.ntest > 0:
            test_samples = test_samples[:args.ntest]
        print(f"Number of test examples: {len(test_samples)}")
        
        # Create test_data dictionary
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
        
        # Create DataFrame
        dev_df = pd.DataFrame(dev_data)
        test_df = pd.DataFrame(test_data)

        # Evaluate
        cors, acc, subject_results = eval(args, subject, llm, dev_df, test_df)
        all_results.extend(subject_results)
        
        # Save result
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        # Save result
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

    # Calculate average correct rate for each category
    for cat in categories:
        cat_subjects = results["categories"][cat]["subjects"]
        if cat_subjects:
            cat_rate = np.mean([s["correct_rate"] for s in cat_subjects])
            results["categories"][cat]["correct_rate"] = round(cat_rate, 2)
            print(f"\nAverage accuracy {cat_rate:.2f} - {cat}")

    # Calculate overall average correct rate
    all_rates = [s["correct_rate"] for s in results["subjects"].values()]
    if all_rates:
        results["overall_correct_rate"] = round(np.mean(all_rates), 2)
        print(f"\nOverall accuracy: {results['overall_correct_rate']:.2f}")

    # Modify results directory structure
    results_dir = os.path.join(args.save_dir, args.model_name.replace('/', '_'))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Save results JSON file
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    save_path = os.path.join(
        results_dir,
        f"results_{timestamp}.json"
    )
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save average correct rate summary file
    total_time = time.time() - start_time
    summary = {
        "timestamp": timestamp,
        "model_name": args.model_name,
        "total_time": f"{total_time:.1f}s",
        "overall_correct_rate": round(results["overall_correct_rate"], 2),
        "category_correct_rates": {
            cat: round(results["categories"][cat]["correct_rate"], 2)
            for cat in categories
        }
    }
    
    summary_path = os.path.join(args.save_dir, "correct_summary.jsonl")
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(summary, ensure_ascii=False) + '\n')
    
    print(f"\nResults saved to: {save_path}")
    print(f"Accuracy summary appended to: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto_mode", "-am", action="store_true",
                      help="Automatically set up using CLI arguments")
    parser.add_argument("--categories", "-c", type=str,
                      help="Categories to evaluate (comma-separated)")
    parser.add_argument("--subjects", "-sb", type=str,
                      help="Subjects to evaluate (comma-separated)")
    parser.add_argument("--use_few_shot", "-f", action="store_true", default=False,
                      help="Use few-shot learning")
    parser.add_argument("--ntrain", "-k", type=int, default=5,
                      help="Number of few-shot examples")
    parser.add_argument("--ntest", "-n", type=int, default=-1,
                      help="Number of test questions per subject")
    parser.add_argument("--nsubjects", "-ns", type=int, default=-1)
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--base_url", type=str, default="http://localhost:1234/v1")
    parser.add_argument("--api_key", type=str, default="lm-studio")
    parser.add_argument("--model_name", type=str, default="llama-3-8b-instruct@?")
    parser.add_argument("--verbose", "-v", action="store_true", 
                      help="Include few-shot examples in output logs")
    args = parser.parse_args()
    main(args) 