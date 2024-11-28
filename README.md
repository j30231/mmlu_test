# MMLU Test Source Code

Evaluate the performance of Language Models (LLMs) using HuggingFace's `cais/mmlu` dataset. This project measures the accuracy of models across various categories and subjects within the MMLU dataset.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Mode Description](#mode-description)
  - [Command Examples](#command-examples)
- [Categories and Subjects](#categories-and-subjects)
- [Parallel Processing Modes](#parallel-processing-modes)
- [Contributing](#contributing)
- [Contact](#contact)

## Installation

Clone the project to your local environment and install the required packages.

```bash:README.md
git clone https://github.com/yourusername/mmlu-test.git
cd mmlu-test
pip install -r requirements.txt
```

## Usage

### Mode Description

- `--auto_mode`: Runs the tests in automatic mode, performing either full or specified range of tests.
- `--ntest`: Specifies the number of questions to test. `-1` signifies a full test.
- `--model_name`: Specifies the name of the model to evaluate.
- `--category`: Evaluates only the subjects within a specific category.
- `-f`: Enables Few-shot learning.
- `-k`: Specifies the number of examples to use in Few-shot learning.

### Command Examples

#### 1. Full Model Evaluation

Perform a full test on the `llama-3-8b-instruct@iq2_s` model.

```bash:mmlu_eval1_sync.py
python mmlu_eval1_sync.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@iq2_s
```

#### 2. Evaluate 10 Questions in the STEM Category

Evaluate 10 questions from the STEM category using the `llama-3-8b-instruct@q4_k_m` model.

```bash:mmlu_eval1_sync.py
python mmlu_eval1_sync.py --auto_mode --ntest -10 --model_name llama-3-8b-instruct@q4_k_m --category 1
```

#### 3. Evaluate 10 Questions in the STEM Category with Few-shot Learning (5 Examples)

Evaluate 10 questions from the STEM category using the `llama-3-8b-instruct@q4_k_m` model with Few-shot learning (5 examples).

```bash:mmlu_eval1_sync.py
python mmlu_eval1_sync.py --auto_mode --ntest -10 --model_name llama-3-8b-instruct@q4_k_m --category 1 -f -k 5
```

#### 4. Direct User Input Mode

Evaluate the model with user-provided input.

```bash:mmlu_eval1_sync.py
python mmlu_eval1_sync.py
```

## Categories and Subjects

<details>
  <summary>View Categories and Subjects</summary>

### List of Categories

1. **STEM** (18 subjects)
   - Abstract Algebra, Astronomy, College Biology, College Chemistry, College Computer Science, College Mathematics, College Physics, Computer Security, Conceptual Physics, Electrical Engineering, Elementary Mathematics, High School Biology, High School Chemistry, High School Computer Science, High School Mathematics, High School Physics, High School Statistics, Machine Learning

2. **Humanities** (13 subjects)
   - Formal Logic, High School European History, High School US History, High School World History, International Law, Jurisprudence, Logical Fallacies, Moral Disputes, Moral Scenarios, Philosophy, Prehistory, Professional Law, World Religions

3. **Social Sciences** (12 subjects)
   - Econometrics, High School Geography, High School Government and Politics, High School Macroeconomics, High School Microeconomics, High School Psychology, Human Sexuality, Professional Psychology, Public Relations, Security Studies, Sociology, US Foreign Policy

4. **Other (Business, Health, Misc.)** (14 subjects)
   - Anatomy, Business Ethics, Clinical Knowledge, College Medicine, Global Facts, Human Aging, Management, Marketing, Medical Genetics, Miscellaneous, Nutrition, Professional Accounting, Professional Medicine, Virology

</details>

### Complete List of Subjects

```plaintext:README.md
1. abstract_algebra
2. anatomy
3. astronomy
...
57. world_religions
```

## Parallel Processing Modes

### Asynchronous Processing

Enhance evaluation speed through parallel processing.

```bash:mmlu_eval2_async.py
python mmlu_eval2_async.py
```

### Asynchronous Processing for Low-Performance Models

Use asynchronous processing for low-performance models, such as 1-bit Quantized Models.

```bash:mmlu_eval3_async.py
python mmlu_eval3_async.py
```

### Batch Execution of All Models

Execute all models in batch using LM Studio with the LMS CLI.

```bash:batch.sh
./batch.sh
```

## Contributing

Contributions are always welcome! Fork this repository, add features or fix bugs, and submit a pull request.

## Contact

For any inquiries, please contact us at: [j30231@gmail.com](mailto:j30231@gmail.com)