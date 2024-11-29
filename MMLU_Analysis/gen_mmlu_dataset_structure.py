from datasets import load_dataset, get_dataset_config_names
import json
from collections import defaultdict
from pprint import pprint

def analyze_dataset_structure():
    """Analyzes the overall structure of the MMLU dataset."""
    
    print("Starting MMLU dataset structure analysis...\n")
    
    try:
        # Get all available configurations
        configs = get_dataset_config_names('cais/mmlu')
        print(f"Found configurations: {configs}\n")
        
        dataset_structure = {
            "configurations": {},
            "metadata": {
                "dataset_name": "cais/mmlu",
                "total_configs": len(configs)
            }
        }
        
        for config in configs:
            print(f"\n[{config}] configuration analysis in progress...")
            try:
                ds = load_dataset('cais/mmlu', config)
                
                # Collect splits information for each configuration
                splits_info = {}
                for split_name, split_data in ds.items():
                    # Collect samples
                    total_samples = len(split_data)
                    
                    # Collect column information
                    columns = split_data.column_names
                    features = {col: str(split_data.features[col]) for col in columns}
                    
                    # Subject distribution analysis (if available)
                    subject_stats = defaultdict(int)
                    if 'subject' in columns:
                        for sample in split_data:
                            if 'subject' in sample:
                                subject_stats[sample['subject']] += 1
                    
                    # Collect sample examples (first sample)
                    example = split_data[0] if total_samples > 0 else None
                    
                    splits_info[split_name] = {
                        "total_samples": total_samples,
                        "columns": columns,
                        "features": features,
                        "subject_distribution": dict(subject_stats) if subject_stats else None,
                        "example": example
                    }
                
                dataset_structure["configurations"][config] = {
                    "splits": splits_info,
                    "total_splits": len(splits_info)
                }
                
                print(f"- Found {len(splits_info)} splits")
                for split_name, split_info in splits_info.items():
                    print(f"  - {split_name}: {split_info['total_samples']} samples")
                    print(f"    Columns: {', '.join(split_info['columns'])}")
                    if split_info['subject_distribution']:
                        print(f"    {len(split_info['subject_distribution'])} unique subjects")
            
            except Exception as e:
                print(f"Warning: Error occurred during loading {config} configuration: {str(e)}")
                dataset_structure["configurations"][config] = {"error": str(e)}
        
        # Calculate overall statistics
        total_samples = 0
        total_subjects = set()
        all_columns = set()
        for config in dataset_structure["configurations"].values():
            if "splits" in config:
                for split_info in config["splits"].values():
                    total_samples += split_info["total_samples"]
                    all_columns.update(split_info["columns"])
                    if split_info["subject_distribution"]:
                        total_subjects.update(split_info["subject_distribution"].keys())
        
        dataset_structure["metadata"].update({
            "total_samples": total_samples,
            "total_unique_subjects": len(total_subjects),
            "all_subjects": sorted(list(total_subjects)),
            "all_columns": sorted(list(all_columns))
        })
        
        # Save results as a JSON file
        with open('mmlu_dataset_structure.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_structure, f, ensure_ascii=False, indent=2)
        
        print("\nAnalysis complete!")
        print(f"Total configurations: {len(configs)}")
        print(f"Total samples: {total_samples}")
        print(f"Total unique subjects: {len(total_subjects)}")
        print(f"All columns: {', '.join(all_columns)}")
        print("\nDetailed analysis results have been saved to 'mmlu_dataset_structure.json'")
        
        return dataset_structure
    
    except Exception as e:
        print(f"Error: Error occurred during dataset analysis: {str(e)}")
        return None

if __name__ == "__main__":
    dataset_structure = analyze_dataset_structure()
    
    if dataset_structure:
        print("\nDataset structure summary:")
        print("=" * 50)
        print(f"Total configurations: {dataset_structure['metadata']['total_configs']}")
        print(f"Total samples: {dataset_structure['metadata']['total_samples']}")
        print(f"Total unique subjects: {dataset_structure['metadata']['total_unique_subjects']}")
        print(f"All columns: {', '.join(dataset_structure['metadata']['all_columns'])}")
        print("\nConfiguration-wise information:")
        for config_name, config_info in dataset_structure["configurations"].items():
            print(f"\n[{config_name}]")
            if "splits" in config_info:
                for split_name, split_info in config_info["splits"].items():
                    print(f"- {split_name}: {split_info['total_samples']} samples")
                    if split_info['subject_distribution']:
                        subjects_count = len(split_info['subject_distribution'])
                        print(f"  ({subjects_count} unique subjects)") 