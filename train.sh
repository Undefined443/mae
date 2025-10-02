export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=24
OUTPUT_DIR="output/origin"

torchrun \
    --nproc-per-node auto \
    main_finetune.py \
    --batch_size 512 \
    --model vit_base_patch16 \
    --epochs 300 \
    --blr 1e-4 \
    --layer_decay 0.75 \
    --weight_decay 0.3 \
    --drop_path 0.1 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --dist_eval \
    --data_path data \
    --accum_iter 1 \
    --warmup_epochs 20 \
    --aa rand-m9-mstd0.5-inc1 \
    --dist_eval \
    --output_dir $OUTPUT_DIR \
    --log_dir $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/train.log
