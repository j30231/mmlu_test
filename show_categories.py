from categories import categories, subcategories

def show_all_categories():
    print("\n=== 카테고리 및 과목 목록 ===")
    
    # 전체 과목 리스트 생성
    all_subjects = []
    category_subjects = {cat: [] for cat in categories.keys()}
    
    # 카테고리별 과목 매핑 생성
    for subject, subcats in subcategories.items():
        for cat, subcat_list in categories.items():
            if any(subcat in subcat_list for subcat in subcats):
                category_subjects[cat].append(subject)
                all_subjects.append((cat, subject))
                break
    
    # 전체 과목 리스트 출력
    print("\n=== 전체 과목 리스트 (자동 모드용) ===")
    for i, (cat, subject) in enumerate(all_subjects, 1):
        print(f"{i}. [{cat}] {subject}")
    
    # 카테고리별 출력
    print("\n=== 카테고리별 과목 목록 (대화형 모드용) ===")
    for i, (category, subjects) in enumerate(category_subjects.items(), 1):
        print(f"\n{i}. {category} ({len(subjects)} subjects)")
        print("   과목 목록:")
        for j, subject in enumerate(sorted(subjects), 1):
            print(f"   {j}. {subject}")

if __name__ == "__main__":
    show_all_categories() 