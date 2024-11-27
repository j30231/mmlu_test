MMLU Test 소스코드

HuggingFace cais/mmlu dataset을 통해서 테스트
LLM Model (API방식) 에 대해서 평가 수행

# 실행 예시

# MMLU 카테고리와 서브젝트를 확인하는 방법
> python show_categories.py

```
=== 카테고리 및 과목 목록 ===

=== 전체 과목 리스트 (자동 모드용) ===
1. [STEM] abstract_algebra
2. [other (business, health, misc.)] anatomy
3. [STEM] astronomy
4. [other (business, health, misc.)] business_ethics
5. [other (business, health, misc.)] clinical_knowledge
6. [STEM] college_biology
7. [STEM] college_chemistry
8. [STEM] college_computer_science
9. [STEM] college_mathematics
10. [other (business, health, misc.)] college_medicine
11. [STEM] college_physics
12. [STEM] computer_security
13. [STEM] conceptual_physics
14. [social sciences] econometrics
15. [STEM] electrical_engineering
16. [STEM] elementary_mathematics
17. [humanities] formal_logic
18. [other (business, health, misc.)] global_facts
19. [STEM] high_school_biology
20. [STEM] high_school_chemistry
21. [STEM] high_school_computer_science
22. [humanities] high_school_european_history
23. [social sciences] high_school_geography
24. [social sciences] high_school_government_and_politics
25. [social sciences] high_school_macroeconomics
26. [STEM] high_school_mathematics
27. [social sciences] high_school_microeconomics
28. [STEM] high_school_physics
29. [social sciences] high_school_psychology
30. [STEM] high_school_statistics
31. [humanities] high_school_us_history
32. [humanities] high_school_world_history
33. [other (business, health, misc.)] human_aging
34. [social sciences] human_sexuality
35. [humanities] international_law
36. [humanities] jurisprudence
37. [humanities] logical_fallacies
38. [STEM] machine_learning
39. [other (business, health, misc.)] management
40. [other (business, health, misc.)] marketing
41. [other (business, health, misc.)] medical_genetics
42. [other (business, health, misc.)] miscellaneous
43. [humanities] moral_disputes
44. [humanities] moral_scenarios
45. [other (business, health, misc.)] nutrition
46. [humanities] philosophy
47. [humanities] prehistory
48. [other (business, health, misc.)] professional_accounting
49. [humanities] professional_law
50. [other (business, health, misc.)] professional_medicine
51. [social sciences] professional_psychology
52. [social sciences] public_relations
53. [social sciences] security_studies
54. [social sciences] sociology
55. [social sciences] us_foreign_policy
56. [other (business, health, misc.)] virology
57. [humanities] world_religions

=== 카테고리별 과목 목록 (대화형 모드용) ===

1. STEM (18 subjects)
   과목 목록:
   1. abstract_algebra
   2. astronomy
   3. college_biology
   4. college_chemistry
   5. college_computer_science
   6. college_mathematics
   7. college_physics
   8. computer_security
   9. conceptual_physics
   10. electrical_engineering
   11. elementary_mathematics
   12. high_school_biology
   13. high_school_chemistry
   14. high_school_computer_science
   15. high_school_mathematics
   16. high_school_physics
   17. high_school_statistics
   18. machine_learning

2. humanities (13 subjects)
   과목 목록:
   1. formal_logic
   2. high_school_european_history
   3. high_school_us_history
   4. high_school_world_history
   5. international_law
   6. jurisprudence
   7. logical_fallacies
   8. moral_disputes
   9. moral_scenarios
   10. philosophy
   11. prehistory
   12. professional_law
   13. world_religions

3. social sciences (12 subjects)
   과목 목록:
   1. econometrics
   2. high_school_geography
   3. high_school_government_and_politics
   4. high_school_macroeconomics
   5. high_school_microeconomics
   6. high_school_psychology
   7. human_sexuality
   8. professional_psychology
   9. public_relations
   10. security_studies
   11. sociology
   12. us_foreign_policy

4. other (business, health, misc.) (14 subjects)
   과목 목록:
   1. anatomy
   2. business_ethics
   3. clinical_knowledge
   4. college_medicine
   5. global_facts
   6. human_aging
   7. management
   8. marketing
   9. medical_genetics
   10. miscellaneous
   11. nutrition
   12. professional_accounting
   13. professional_medicine
   14. virology
```

# arg 변수로 설정을 받는 모드
# Model (llama-3-8b-instruct@iq2_s) 에 대해서 전체 평가 수행
> python mmlu_eval1.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@iq2_s

# Model (llama-3-8b-instruct@q4_k_m) 에 대해서 STEM 카테고리 10개 문항만 평가 수행
> python mmlu_eval1.py --auto_mode --ntest -10 --model_name llama-3-8b-instruct@q4_k_m -category 1

# Model (llama-3-8b-instruct@q4_k_m) 에 대해서 STEM 카테고리 10개 문항만 평가 수행, Few-shot 사용(5회)
> python mmlu_eval1.py --auto_mode --ntest -10 --model_name llama-3-8b-instruct@q4_k_m -category 1 -f -k 5

# 사용자 입력을 직접 받는 모드
> python mmlu_eval1.py 

# 병럴 처리모드로 실행할 경우 mmlu_eval2.py 또는 mmlu_evla3.py 사용 (2개 파일은 동일함)
> python mmlu_eval2.py 

Contact E-mail : j30231@gmail.com
