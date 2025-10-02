export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=24

python main_finetune.py \
    --eval \
    --resume output/checkpoint-250.pth \
    --model vit_base_patch16 \
    --batch_size 1024 \
    --data_path data \
    --output_dir temp
