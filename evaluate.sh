export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12

python main_finetune.py \
    --eval \
    --resume output/checkpoint-299.pth \
    --model vit_base_patch16 \
    --batch_size 256 \
    --data_path data \
    --output_dir temp
