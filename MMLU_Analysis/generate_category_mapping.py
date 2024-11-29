from datasets import load_dataset, get_dataset_config_names
import json
from collections import defaultdict
from pprint import pprint

def analyze_dataset_structure():
    """MMLU 데이터셋의 전체 구조를 분석합니다."""
    
    print("MMLU 데이터셋 구조 분석 시작...\n")
    
    try:
        # 사용 가능한 모든 configuration 가져오기
        configs = get_dataset_config_names('cais/mmlu')
        print(f"발견된 configuration 목록: {configs}\n")
        
        dataset_structure = {
            "configurations": {},
            "metadata": {
                "dataset_name": "cais/mmlu",
                "total_configs": len(configs)
            }
        }
        
        for config in configs:
            print(f"\n[{config}] configuration 분석 중...")
            try:
                ds = load_dataset('cais/mmlu', config)
                
                # 각 configuration의 splits 정보 수집
                splits_info = {}
                for split_name, split_data in ds.items():
                    # 샘플 수집
                    total_samples = len(split_data)
                    
                    # 컬럼(특성) 정보 수집
                    columns = split_data.column_names
                    features = {col: str(split_data.features[col]) for col in columns}
                    
                    # subject 분포 분석 (있는 경우)
                    subject_stats = defaultdict(int)
                    if 'subject' in columns:
                        for sample in split_data:
                            if 'subject' in sample:
                                subject_stats[sample['subject']] += 1
                    
                    # 샘플 예시 수집 (첫 번째 샘플)
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
                
                print(f"- {len(splits_info)}개의 splits 발견")
                for split_name, split_info in splits_info.items():
                    print(f"  - {split_name}: {split_info['total_samples']} samples")
                    print(f"    컬럼: {', '.join(split_info['columns'])}")
                    if split_info['subject_distribution']:
                        print(f"    {len(split_info['subject_distribution'])}개의 unique subjects")
            
            except Exception as e:
                print(f"Warning: {config} configuration 로드 중 오류 발생: {str(e)}")
                dataset_structure["configurations"][config] = {"error": str(e)}
        
        # 전체 통계 계산
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
        
        # 결과를 JSON 파일로 저장
        with open('mmlu_dataset_structure.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_structure, f, ensure_ascii=False, indent=2)
        
        print("\n분석 완료!")
        print(f"총 configuration 수: {len(configs)}")
        print(f"총 샘플 수: {total_samples}")
        print(f"총 고유 과목 수: {len(total_subjects)}")
        print(f"모든 컬럼: {', '.join(all_columns)}")
        print("\n상세 분석 결과가 'mmlu_dataset_structure.json'에 저장되었습니다.")
        
        return dataset_structure
    
    except Exception as e:
        print(f"Error: 데이터셋 분석 중 오류 발생: {str(e)}")
        return None

if __name__ == "__main__":
    dataset_structure = analyze_dataset_structure()
    
    if dataset_structure:
        print("\n데이터셋 구조 요약:")
        print("=" * 50)
        print(f"총 configuration 수: {dataset_structure['metadata']['total_configs']}")
        print(f"총 샘플 수: {dataset_structure['metadata']['total_samples']}")
        print(f"총 고유 과목 수: {dataset_structure['metadata']['total_unique_subjects']}")
        print(f"사용 가능한 모든 컬럼: {', '.join(dataset_structure['metadata']['all_columns'])}")
        print("\n각 configuration 별 정보:")
        for config_name, config_info in dataset_structure["configurations"].items():
            print(f"\n[{config_name}]")
            if "splits" in config_info:
                for split_name, split_info in config_info["splits"].items():
                    print(f"- {split_name}: {split_info['total_samples']} samples")
                    if split_info['subject_distribution']:
                        subjects_count = len(split_info['subject_distribution'])
                        print(f"  ({subjects_count} unique subjects)") 