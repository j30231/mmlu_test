from datasets import load_dataset
import json
from collections import defaultdict

def analyze_dataset_structure():
    """Analyzes the overall structure of the MMLU dataset and saves related information.
    Returns:
        json file: mmlu_category_mapping.json (category mapping information)
        json file: mmlu_dataset_info.json (dataset information)
    """
    
    print("Starting MMLU dataset structure analysis...\n")
    
    try:
        # Load dataset
        ds = load_dataset('cais/mmlu', 'all')
        
        # Dataset basic structure information
        dataset_types = {
            "test": "MMLU test data for evaluation",
            "validation": "Data for model validation",
            "dev": "Sample data for few-shot prompts",
            "auxiliary_train": "Data for model training"
        }
        
        # Initialize dataset structure information
        dataset_info = {
            "metadata": {
                "dataset_name": "cais/mmlu",
                "dataset_types": dataset_types
            },
            "data": {}
        }
        
        # Collect information for each data type
        for data_type in dataset_types.keys():
            if data_type in ds:
                split_data = ds[data_type]
                subjects = set()
                subject_stats = defaultdict(int)
                
                # Collect statistics for each subject
                for sample in split_data:
                    if 'subject' in sample and sample['subject'].strip():
                        subjects.add(sample['subject'])
                        subject_stats[sample['subject']] += 1
                
                dataset_info["data"][data_type] = {
                    "description": dataset_types[data_type],
                    "total_samples": len(split_data),
                    "columns": split_data.column_names,
                    "unique_subjects": len(subjects),
                    "subjects": sorted(list(subjects)),
                    "subject_stats": dict(subject_stats),
                    "example": split_data[0] if len(split_data) > 0 else None
                }
        
        # Collect total subjects and statistics
        all_subjects = set()
        for data_type_info in dataset_info["data"].values():
            all_subjects.update(data_type_info["subjects"])
        
        dataset_info["metadata"]["total_subjects"] = len(all_subjects)
        dataset_info["metadata"]["all_subjects"] = sorted(list(all_subjects))
        
        # Create category mapping information
        category_mapping = {
            "metadata": {
                "description": "Category mapping information for the MMLU dataset"
            },
            "mappings": {}
        }
        
        # Create mapping information for each data type
        for data_type in dataset_types.keys():
            if data_type in dataset_info["data"]:
                category_mapping["mappings"][data_type] = {
                    "description": dataset_types[data_type],
                    "subjects": dataset_info["data"][data_type]["subjects"],
                    "total_samples": dataset_info["data"][data_type]["total_samples"],
                    "subject_distribution": dataset_info["data"][data_type]["subject_stats"]
                }
        
        # Save results
        with open('mmlu_dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
            
        with open('mmlu_category_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(category_mapping, f, ensure_ascii=False, indent=2)
        
        print("\nAnalysis completed!")
        print(f"Total subjects: {len(all_subjects)}")
        for data_type, info in dataset_info["data"].items():
            print(f"\n{data_type} ({dataset_types[data_type]}):")
            print(f"  - Total samples: {info['total_samples']}")
            print(f"  - Unique subjects: {info['unique_subjects']}")
            print(f"  - Columns: {', '.join(info['columns'])}")
        
        print("\nAnalysis results are saved in the following files:")
        print("- mmlu_dataset_info.json")
        print("- mmlu_category_mapping.json")
        
        return dataset_info, category_mapping
    
    except Exception as e:
        print(f"Error: Error occurred during dataset analysis: {str(e)}")
        return None, None

if __name__ == "__main__":
    dataset_info, category_mapping = analyze_dataset_structure() 