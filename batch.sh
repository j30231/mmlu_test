#!/bin/bash

# model names for API calls with LM Studio (http://localhost:1234)
# "llama3-8b-iq1s": "llama-3-8b-instruct@iq1_s",
# "llama3-8b-iq1m": "llama-3-8b-instruct@iq1_m",
# "llama3-8b-iq2xss": "llama-3-8b-instruct@iq2_xss",
# "llama3-8b-q2": "llama-3-8b-instruct@q2_k",
# "llama3-8b-q2ks": "llama-3-8b-instruct@q2_k_s",
# "llama3-8b-iq2s": "llama-3-8b-instruct@iq2_s",
# "llama3-8b-iq2m": "llama-3-8b-instruct@iq2_m",
# "llama3-8b-iq2xs": "llama-3-8b-instruct@iq2_xs",
# "llama3-8b-q3": "llama-3-8b-instruct@q3_k_l",
# "llama3-8b-q3ks": "llama-3-8b-instruct@q3_k_s",
# "llama3-8b-q4": "llama-3-8b-instruct@q4_0",
# "llama3-8b-iq4nl": "llama-3-8b-instruct@iq4_nl",
# "llama3-8b-q4km": "llama-3-8b-instruct@q4_k_m",
# "llama3-8b-q5km": "llama-3-8b-instruct@q5_k_m",
# "llama3-8b-q8": "llama-3-8b-instruct@q8_0",
# "llama3-8b-q16": "llama-3-8b-instruct-q16_0",
# "llama3-8b-f16": "llama-3-8b-instruct@?",

# lms load "meta/Llama-3-8B-Instruct/"
# python -u mmlu_eval2.py --auto_mode --ntest -1 --model_name "llama-3-8b-instruct@?" 2>&1 | tee -a results/f16.txt
# lms unload "llama-3-8b-instruct@?"

# lms load "makeself/Llama-3-8B-Instruct-q16_0/Meta-Llama-3-8B-Instruct.gguf"
# python -u mmlu_eval2.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct-q16_0 2>&1 | tee -a results/q16_0.txt
# lms unload "llama-3-8b-instruct-q16_0"

# lms load "makeself/Llama-3-8B-Instruct-q8_0/Meta-Llama-3-8B-Instruct-q8_0.gguf"
# python -u mmlu_eval2.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@q8_0 2>&1 | tee -a results/q8_0.txt
# lms unload "llama-3-8b-instruct@q8_0"

# lms load "makeself/Llama-3-8B-Instruct-Q5_K_M/llama-3-8b-instruct-Q5_K_M.gguf"
# python -u mmlu_eval2.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@q5_k_m 2>&1 | tee -a results/q5_k_m.txt
# lms unload "llama-3-8b-instruct@q5_k_m"

# lms load "makeself/llama-3-8b-instruct-IQ4_NL/llama-3-8b-instruct-IQ4_NL.gguf"
# python -u mmlu_eval2.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@iq4_nl 2>&1 | tee -a results/iq4_nl.txt
# lms unload "llama-3-8b-instruct@iq4_nl"

# lms load "makeself/Llama-3-8B-Instruct-Q4_K_M/llama-3-8b-instruct-Q4_K_M.gguf"
# python -u mmlu_eval2.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@q4_k_m 2>&1 | tee -a results/q4_k_m.txt
# lms unload "llama-3-8b-instruct@q4_k_m"

# lms load "makeself/Llama-3-8B-Instruct-q4_0/Meta-Llama-3-8B-Instruct-q4_0.gguf"
# python -u mmlu_eval2.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@q4_0 2>&1 | tee -a results/q4_0.txt
# lms unload "llama-3-8b-instruct@q4_0"

# lms load "makeself/Llama-3-8B-Instruct-q3_k_l/Meta-Llama-3-8B-Instruct-q3_k_l.gguf"
# python -u mmlu_eval2.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@q3_k_l 2>&1 | tee -a results/q3_k_l.txt
# lms unload "llama-3-8b-instruct@q3_k_l"

# lms load "makeself/Llama-3-8B-Instruct-Q3_K_S/llama-3-8b-instruct-Q3_K_S.gguf"
# python -u mmlu_eval2.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@q3_k_s 2>&1 | tee -a results/q3_k_s.txt
# lms unload "llama-3-8b-instruct@q3_k_s"

# lms load "makeself/Llama-3-8B-Instruct-q2_k_s/llama-3-8b-instruct-q2_k_s.gguf"
# python -u mmlu_eval2.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@q2_k_s 2>&1 | tee -a results/q2_k_s.txt
# lms unload "llama-3-8b-instruct@q2_k_s"

# lms load "makeself/Llama-3-8B-Instruct-IQ1_M/llama-3-8b-instruct-IQ1_M.gguf"
# python -u mmlu_eval3_async.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@iq1_m 2>&1 | tee -a results/iq1_m.txt
# lms unload "llama-3-8b-instruct@iq1_m"

# lms load "makeself/Llama-3-8B-Instruct-IQ1_S/llama-3-8b-instruct-IQ1_S.gguf"
# python -u mmlu_eval3_async.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@iq1_s 2>&1 | tee -a results/iq1_s.txt
# lms unload "llama-3-8b-instruct@iq1_s"
