export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=24
OUTPUT_DIR="output/origin"

python main_finetune.py \
    --eval \
    --resume $OUTPUT_DIR/checkpoint-250.pth \
    --model vit_base_patch16 \
    --batch_size 1024 \
    --data_path data \
    --output_dir temp \
    --log_dir temp \
    2>&1 | tee $OUTPUT_DIR/evaluate.log
