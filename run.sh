# conda create -n paddlenlp python=3.7
# source activate paddlenlp
# pip install =r requirements.txt

python process_data.py preprocess

python3 run_seq2struct.py                              \
  --multi_task_config config/multi-task-duuie-local.yaml     \
  --negative_keep 1.0                                  \
  --do_train                                           \
  --metric_for_best_model=all-task-ave                 \
  --model_name_or_path=./uie-char-small                \
  --num_train_epochs=10                                \
  --per_device_train_batch_size=16                     \
  --per_device_eval_batch_size=256                     \
  --output_dir=output/duuie_multi_task_b32_lr5e-4      \
  --logging_dir=output/duuie_multi_task_b32_lr5e-4_log \
  --learning_rate=5e-4                                 \
  --overwrite_output_dir                               \
  --gradient_accumulation_steps 1                      \
  --device gpu

#   --model_name_or_path=./duuie_multi_task_b32_lr5e-4/ckpt_epoch9                \

python process_data.py split-test
python inference.py --data data/duuie_test_a --model output/duuie_multi_task_b32_lr5e-4

python process_data.py merge-test


