# MMLU Categories
MMLU_CATEGORIES = [
    "STEM",
    "Humanities",
    "Social Sciences",
    "Other"
]

# MMLU Subjects
MMLU_SUBJECTS = [
    "Abstract Algebra", "Anatomy", "Astronomy", "Business Ethics", "Clinical Knowledge", "College Biology",
    "College Chemistry", "College Computer Science", "College Mathematics", "College Medicine", "College Physics",
    "Computer Security", "Conceptual Physics", "Econometrics", "Electrical Engineering", "Elementary Mathematics",
    "Formal Logic", "Global Facts", "High School Biology", "High School Chemistry", "High School Computer Science",
    "High School European History", "High School Geography", "High School Government and Politics",
    "High School Macroeconomics", "High School Mathematics", "High School Microeconomics", "High School Physics",
    "High School Psychology", "High School Statistics", "High School US History", "High School World History",
    "Human Aging", "Human Sexuality", "International Law", "Jurisprudence", "Logical Fallacies", "Machine Learning",
    "Management", "Marketing", "Medical Genetics", "Miscellaneous", "Moral Disputes", "Moral Scenarios",
    "Nutrition", "Philosophy", "Prehistory", "Professional Accounting", "Professional Law", "Professional Medicine",
    "Professional Psychology", "Public Relations", "Security Studies", "Sociology", "US Foreign Policy",
    "Virology", "World Religions"
]

# Subject to Category Mapping (데이터셋의 subject 필드와 일치하도록 수정)
SUBJECT_TO_CATEGORY = {
    "abstract_algebra": "STEM",
    "anatomy": "STEM",
    "astronomy": "STEM",
    "business_ethics": "Humanities",
    "clinical_knowledge": "STEM",
    "college_biology": "STEM",
    "college_chemistry": "STEM",
    "college_computer_science": "STEM",
    "college_mathematics": "STEM",
    "college_medicine": "STEM",
    "college_physics": "STEM",
    "computer_security": "STEM",
    "conceptual_physics": "STEM",
    "econometrics": "Social Sciences",
    "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM",
    "formal_logic": "Humanities",
    "global_facts": "Social Sciences",
    "high_school_biology": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM",
    "high_school_european_history": "Humanities",
    "high_school_geography": "Social Sciences",
    "high_school_government_and_politics": "Social Sciences",
    "high_school_macroeconomics": "Social Sciences",
    "high_school_mathematics": "STEM",
    "high_school_microeconomics": "Social Sciences",
    "high_school_physics": "STEM",
    "high_school_psychology": "Social Sciences",
    "high_school_statistics": "STEM",
    "high_school_us_history": "Humanities",
    "high_school_world_history": "Humanities",
    "human_aging": "Social Sciences",
    "human_sexuality": "Social Sciences",
    "international_law": "Humanities",
    "jurisprudence": "Humanities",
    "logical_fallacies": "Humanities",
    "machine_learning": "STEM",
    "management": "Social Sciences",
    "marketing": "Social Sciences",
    "medical_genetics": "STEM",
    "miscellaneous": "Other",
    "moral_disputes": "Humanities",
    "moral_scenarios": "Humanities",
    "nutrition": "STEM",
    "philosophy": "Humanities",
    "prehistory": "Humanities",
    "professional_accounting": "Social Sciences",
    "professional_law": "Humanities",
    "professional_medicine": "STEM",
    "professional_psychology": "Social Sciences",
    "public_relations": "Social Sciences",
    "security_studies": "Social Sciences",
    "sociology": "Social Sciences",
    "us_foreign_policy": "Social Sciences",
    "virology": "STEM",
    "world_religions": "Humanities"
}

# MMLU Dataset Information
MMLU_DATASET_INFO = {
    "total_subjects": 57,
    "minimum_examples_per_subject": 100,
    "question_format": "multiple-choice",
    "difficulty_range": ["elementary school level", "middle school level", "high school level", "college level", "professional level"],
    "evaluation_settings": ["zero-shot", "few-shot"]
}

# MMLU Evaluation Features
MMLU_EVALUATION_FEATURES = [
    "Covers a wide range of subjects with in-depth content",
    "Evaluates both world knowledge and problem-solving abilities of models",
    "Useful for identifying model weaknesses",
    "Assesses general knowledge as well as reasoning capabilities"
]

# MMLU Dataset Splits
MMLU_DATASET_SPLITS = ["train", "validation", "test"]

# MMLU Question Structure
MMLU_QUESTION_STRUCTURE = {
    "question": "question text",
    "choices": ["choice A", "choice B", "choice C", "choice D"],
    "answer": "correct answer index (0-3)",
    "subject": "subject of the question"
}
