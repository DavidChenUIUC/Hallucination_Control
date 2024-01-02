echo "|- Using CUDA: $CUDA_VISIBLE_DEVICES"

for model in "./lora-flan-t5-xxl_lr_1.1e-3_bs_32_warmup_0.1_cosinewithrestarts_weightdecay_0.1_maxgradnorm_1_seed_1234/checkpoint-2305"
do
    echo "|- Evaluating model $model"
    python3 flan_t5.py --eval_only True --resume_from_checkpoint $model --metric bertscore
done

# echo "Running: ./results_lora-flan-t5-xxl-5e-4_adamw_seed1234"
# python3 flan_t5.py --eval_only True --resume_from_checkpoint "./backup_ckpt/results_lora-flan-t5-xxl-5e-4_adamw_seed1234" --metric rouge

# echo "running: ./backup_ckpt/results_lr1e-3_seed1234"
# python3 flan_t5.py --eval_only True --resume_from_checkpoint "./backup_ckpt/results_lr1e-3_seed1234" --metric rouge

# echo "|- Evaluating base model"
# python3 flan_t5.py --eval_only True --metric rouge
# python3 flan_t5.py --eval_only True --metric bertscore

