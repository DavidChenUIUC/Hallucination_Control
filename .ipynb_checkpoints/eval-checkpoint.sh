echo "|- Using CUDA: $CUDA_VISIBLE_DEVICES"

for model in " ./backup_ckpt/results_lora-flan-t5-xxl-1e-3_linear_warm0.1_seed1234" " ./backup_ckpt/results_lora-flan-t5-xxl-5e-4_adamw_seed1234" " ./backup_ckpt/results_lr1e-3_seed1234"
do
    echo "|- Evaluating model $model"
    python3 flan_t5.py --eval_only True --resume_from_checkpoint $model --metric bertscore
done

echo "|- Evaluating base model"
python3 flan_t5.py --eval_only True --metric bertscore

