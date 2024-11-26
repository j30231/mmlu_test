from datasets import load_dataset
import json
from collections import defaultdict

def analyze_dataset_structure():
    """MMLU 데이터셋의 전체 구조를 분석하고 관련 정보를 저장합니다.
    Returns:
        json file: mmlu_category_mapping.json (카테고리 매핑 정보)
        json file: mmlu_dataset_info.json (데이터셋 정보)
    """
    
    print("MMLU 데이터셋 구조 분석 시작...\n")
    
    try:
        # 데이터셋 로드
        ds = load_dataset('cais/mmlu', 'all')
        
        # 데이터셋 기본 구조 정보
        dataset_types = {
            "test": "MMLU 평가용 테스트 데이터",
            "validation": "모델 검증용 데이터",
            "dev": "Few-shot 프롬프트용 샘플 데이터",
            "auxiliary_train": "모델 학습용 데이터"
        }
        
        # 데이터셋 구조 정보 초기화
        dataset_info = {
            "metadata": {
                "dataset_name": "cais/mmlu",
                "dataset_types": dataset_types
            },
            "data": {}
        }
        
        # 각 데이터 타입별 정보 수집
        for data_type in dataset_types.keys():
            if data_type in ds:
                split_data = ds[data_type]
                subjects = set()
                subject_stats = defaultdict(int)
                
                # 과목별 통계 수집
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
        
        # 전체 과목 목록 및 통계
        all_subjects = set()
        for data_type_info in dataset_info["data"].values():
            all_subjects.update(data_type_info["subjects"])
        
        dataset_info["metadata"]["total_subjects"] = len(all_subjects)
        dataset_info["metadata"]["all_subjects"] = sorted(list(all_subjects))
        
        # 카테고리 매핑 정보 생성
        category_mapping = {
            "metadata": {
                "description": "MMLU 데이터셋의 분류체계별 매핑 정보"
            },
            "mappings": {}
        }
        
        # 각 데이터 타입별 매핑 정보 생성
        for data_type in dataset_types.keys():
            if data_type in dataset_info["data"]:
                category_mapping["mappings"][data_type] = {
                    "description": dataset_types[data_type],
                    "subjects": dataset_info["data"][data_type]["subjects"],
                    "total_samples": dataset_info["data"][data_type]["total_samples"],
                    "subject_distribution": dataset_info["data"][data_type]["subject_stats"]
                }
        
        # 결과 저장
        with open('mmlu_dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
            
        with open('mmlu_category_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(category_mapping, f, ensure_ascii=False, indent=2)
        
        print("\n분석 완료!")
        print(f"총 과목 수: {len(all_subjects)}")
        for data_type, info in dataset_info["data"].items():
            print(f"\n{data_type} ({dataset_types[data_type]}):")
            print(f"  - 샘플 수: {info['total_samples']}")
            print(f"  - 고유 과목 수: {info['unique_subjects']}")
            print(f"  - 컬럼: {', '.join(info['columns'])}")
        
        print("\n분석 결과가 다음 파일들에 저장되었습니다:")
        print("- mmlu_dataset_info.json")
        print("- mmlu_category_mapping.json")
        
        return dataset_info, category_mapping
    
    except Exception as e:
        print(f"Error: 데이터셋 분석 중 오류 발생: {str(e)}")
        return None, None

if __name__ == "__main__":
    dataset_info, category_mapping = analyze_dataset_structure() 