## Hallucination Detection and reduction

Presentation Slides: [[link]](https://drive.google.com/file/d/159BeKL2M6bOtvD6ZTDe18MxipKpm27hv/view?usp=sharing)

## Overview
This repository contains scripts to PEFT model to text summarization task and pipeline for hallucination detection and reduction. 

- `flan_t5.py`: PEFT LoRA-Flan-T5-XXXL on the SamSum dataset.
- `qa_gpt.py`: QA agent utilizing GPT.
- `random_forest.py`: Train or run inference with the Random Forest model.
- `random_sampled_SamSum.csv`: Randomly sampled dataset from SamSum for analysis.
- `Reduced_SamSum_hallucination_summaries.csv`: Dataset with summaries that have been evaluated for hallucination content.
- `samsum_dataset_hal_detection.py`: Detecting hallucinations in the SamSum dataset.
- `samsum_hal_detection.py`:  Detecting hallucinations in generated summaries for SamSum.
- `samsum_hal_reduction.py`: Teacher-Student-like hallucination reduction pipeline to reduce hallucinations in generated summaries.
- `SamSum_test_set_eval_gpt3.csv`: Evaluation annotations of GPT-3 on the SamSum test set.
- `xsum_hal_detection.py`: Detecting hallucinations in the XSum dataset.
- `xsum_hallucination_gpt3_prediction.csv`: Predictions of hallucinations by GPT-3 on the XSum dataset.

