python -u mmlu_eval1.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@iq2_m 2>&1 | tee -a results/iq2_m.txt
python -u mmlu_eval2.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@iq2_s 2>&1 | tee -a results/iq2_s.txt
python -ummlu_eval1.py --auto_mode --ntest -1 --model_name llama-3-8b-instruct@q4_k_m 2>&1 | tee -a results/q4_k_m.txt