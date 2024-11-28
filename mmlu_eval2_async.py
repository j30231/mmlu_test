# Async version (Processing 10 queries in parallel)

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

# Basic settings
choices = ["A", "B", "C", "D"]

# subject_to_category dictionary creation
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
    
    # Find "ANSWER:" or "THE ANSWER IS" pattern
    for line in lines:
        if "ANSWER:" in line or "THE ANSWER IS" in line:
            # Find A, B, C, D in the line
            for char in line:
                if char in ['A', 'B', 'C', 'D']:
                    return char
    
    # If not found by the above method, find the first A, B, C, D in the entire text
    for char in response_text.upper():
        if char in ['A', 'B', 'C', 'D']:
            return char
            
    raise ValueError(f"Could not find a valid answer (A,B,C,D) in: {response_text[:100]}...")

async def async_evaluate_batch(client, messages_batch):
    """Performs asynchronous evaluation in batch mode."""
    async def process_single_message(messages):
        try:
            response = await client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                temperature=0,
                max_tokens=10  # Limit response length
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
    
    # System message modification
    system_msg = """You are a helpful assistant that answers multiple choice questions. 
    Answer with ONLY a single letter (A, B, C, or D) without any explanation."""
    
    # Batch size setting
    BATCH_SIZE = 10  # Number of queries to process at once
    
    # Prepare message batches
    messages_batch = []
    batch_indices = []
    
    for idx, row in test_df.iterrows():
        # Prepare few-shot examples
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
        
        # Process when the batch is full or the last item
        if len(messages_batch) == BATCH_SIZE or idx == len(test_df) - 1:
            start_time = time.time()
            
            # Perform batch evaluation
            responses = await async_evaluate_batch(client, messages_batch)
            
            # Process results
            for batch_idx, (response, test_idx) in enumerate(zip(responses, batch_indices)):
                try:
                    if isinstance(response, Exception):
                        print(f"Error processing question {test_idx}: {str(response)}")
                        continue
                        
                    model_response = response.content.strip()
                    model_answer = extract_answer(model_response)
                    
                    if model_answer is None:
                        print(f"Warning: Could not extract answer from model response: {model_response}")
                        continue
                        
                    model_answer_num = ord(model_answer) - ord('A')
                    correct_answer = test_df.iloc[test_idx]['answer']
                    is_correct = model_answer_num == correct_answer
                    
                    if is_correct:
                        correct_count += 1
                    total_count += 1
                    
                    cors.append(is_correct)
                    accuracy = (correct_count / total_count) * 100
                    
                    # Save result
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
            
            # Reset batch
            messages_batch = []
            batch_indices = []
    
    # Process results
    if total_count == 0:
        print(f"\nWarning: All questions for {subject} failed to process.")
        return np.array([]), 0.0, evaluation_results
    
    return np.array(cors), (correct_count / total_count) * 100, evaluation_results

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
    # Start time recording and timestamp creation
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    
    # Set up directory structure for results
    results_dir = os.path.join(args.save_dir, args.model_name.replace('/', '_'))
    interim_dir = os.path.join(results_dir, f"interim_{timestamp}")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(interim_dir):
        os.makedirs(interim_dir)

    # Initialize results dictionary
    results = {
        "subjects": {},
        "categories": {
            cat: {"subjects": [], "correct_rate": 0.0} 
            for cat in categories
        }
    }

    if args.auto_mode:
        # CLI mode
        category_list = list(categories.keys())
        
        # Category processing
        if args.categories:
            try:
                cat_indices = [int(i.strip()) for i in args.categories.split(',')]
                selected_categories = [category_list[i-1] for i in cat_indices 
                                    if 1 <= i <= len(category_list)]
                # Filter subjects belonging to selected categories
                selected_subjects = [
                    subject for subject, cat in subject_to_category.items()
                    if cat in selected_categories
                ]
                
                # Select subjects by subject index
                if args.subjects:
                    subject_indices = [int(i.strip()) for i in args.subjects.split(',')]
                    selected_subjects = [subject for i, subject in enumerate(selected_subjects, 1)
                                      if i in subject_indices]
                
                # Print few-shot settings
                print(f"\n=== Evaluation Settings ===")
                print(f"Selected categories: {', '.join(selected_categories)}")
                print(f"Selected subjects ({len(selected_subjects)}): {', '.join(selected_subjects)}")
                print(f"Use few-shot: {args.use_few_shot}")
                print(f"Test questions: {'all' if args.ntest <= 0 else args.ntest}")
                print("=" * 30 + "\n")
                
            except (ValueError, IndexError):
                print("Invalid category number. Use numbers between 1 and 4.")
                return
        else:
            selected_categories = category_list
            selected_subjects = list(subject_to_category.keys())
    else:
        # Interactive mode
        selected_subjects, use_few_shot, ntrain, ntest = get_interactive_selections()
        args.use_few_shot = use_few_shot
        args.ntrain = ntrain
        args.ntest = ntest
        # Find categories for selected subjects
        selected_categories = list(set(subject_to_category[subject] for subject in selected_subjects))

    # Remaining code
    if args.nsubjects > 0:
        selected_subjects = random.sample(selected_subjects, min(args.nsubjects, len(selected_subjects)))

    # Define NumPy type conversion function
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Initialize OpenAI client
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key
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
        
        # Prepare development data - adjust based on few-shot setting
        if args.use_few_shot and args.ntrain > 0:
            dev_samples = [x for x in mmlu_dataset['dev'] if x['subject'] == subject][:args.ntrain]
        else:
            dev_samples = []  # If not using few-shot, set to empty list
            
        print(f"Number of few-shot examples: {len(dev_samples)}")  # Message change
        
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

        # Perform evaluation
        cors, acc, subject_results = asyncio.run(async_eval(args, subject, client, dev_df, test_df))
        
        # Handle empty results
        if len(cors) == 0:
            print(f"Warning: No evaluation results for {subject}. Proceeding to the next subject.")
            continue
            
        all_results.extend(subject_results)
        
        # Save results
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        # Save results
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

        # Save intermediate results after each subject
        interim_results = {
            "metadata": {
                "model_name": args.model_name,
                "ntrain": args.ntrain,
                "ntest": args.ntest,
                "timestamp": time.strftime("%Y%m%d-%H%M%S")
            },
            "subject": subject,
            "category": category,
            "correct_rate": float(acc),  # Convert NumPy float to Python float
            "evaluation_results": [
                {
                    "question": result["question"],
                    "model_answer": result["model_answer"],
                    "correct_answer": int(result["correct_answer"]),  # Convert NumPy int to Python int
                    "is_correct": bool(result["is_correct"]),  # Convert NumPy bool to Python bool
                    "response_time": float(result["response_time"])  # Convert NumPy float to Python float
                }
                for result in subject_results
            ]
        }
        
        # Save intermediate results to file
        interim_save_path = os.path.join(
            interim_dir,
            f"interim_{subject}.json"
        )
        
        with open(interim_save_path, 'w', encoding='utf-8') as f:
            json.dump(interim_results, f, ensure_ascii=False, indent=2)
        
        print(f"\nIntermediate results saved to: {interim_save_path}")

    # Calculate average correct rate for each category
    for cat in categories:
        cat_subjects = results["categories"][cat]["subjects"]
        if cat_subjects:
            cat_rate = np.mean([s["correct_rate"] for s in cat_subjects])
            results["categories"][cat]["correct_rate"] = round(cat_rate, 2)
            print(f"\nAverage correct rate: {cat_rate:.2f} - {cat}")

    # Calculate overall average correct rate
    all_rates = [s["correct_rate"] for s in results["subjects"].values()]
    if all_rates:
        results["overall_correct_rate"] = round(np.mean(all_rates), 2)
        print(f"\nOverall correct rate: {results['overall_correct_rate']:.2f}")

    # Calculate total_time before saving final results
    total_time = time.time() - start_time
    
    # Prepare final results before saving
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

    # Organize results by category
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

    # Save final results to JSON file
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    save_path = os.path.join(
        results_dir,
        f"results_{timestamp}.json"
    )
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # Save accuracy summary to file
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
    parser.add_argument("--ntrain", "-k", type=int, default=0,
                      help="Number of few-shot examples")
    parser.add_argument("--ntest", "-n", type=int, default=-1,
                      help="Number of test questions per subject")
    parser.add_argument("--nsubjects", "-ns", type=int, default=-1)
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--base_url", type=str, default="http://localhost:1234/v1")
    parser.add_argument("--api_key", type=str, default="lm-studio")
    parser.add_argument("--model_name", type=str, default="llama-3-8b-instruct@?")
    parser.add_argument("--verbose", "-v", action="store_true", 
                      help="Include few-shot examples in output")
    args = parser.parse_args()
    main(args)
