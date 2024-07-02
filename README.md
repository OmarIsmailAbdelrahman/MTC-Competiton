# MTC-Competition

## Experiments

### 1. Conformer CTC on NeMo
- Achieved the best and most stable results compared to Wav2Vec2.
- Overfitted in the normal configuration at 25 epochs; fine-tuned for an additional 10 epochs.
- Utilized FAdam for optimization.

<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/73082049/2013718c-4a09-49d1-af5e-d3dd6ec0f8bb" alt="Conformer CTC on NeMo"/>
</p>

### 2. Wav2Vec2 Pretraining
- Pretrained Wav2Vec2 but encountered instability, leading to NaN values.
- Instability occurred after 100 epochs with FAdam and 300 epochs with WAdam.

<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/73082049/00660b8d-7726-479c-9102-9d3f7eb3e865" alt="Wav2Vec2 Pretraining"/>
</p>

### 3. KenLM N-gram
- Created multiple n-grams (3 to 6) using the train/adapt dataset, resulting in small output changes.
- Constructed n-grams using the Egyptian Datasets Collection (2.5 million rows), improving performance slightly.

### 4. Knowledge Distillation Model
- Attempted knowledge distillation with Wav2Vec2 and NeMo but failed due to mismatched decoder lengths.
- Potential solution: Adjust transformer heads to match the output.

### 5. FAdam Optimization
- Used FAdam to reduce training time and enhance model stability.
- Demonstrated better results with NeMo CTC Conformer, reaching 35 epochs before overfitting, and 25 epochs in normal training.
- Reference "https://arxiv.org/abs/2405.12807"

### 6. Tokenizer Experiments
- Tried different tokenizers; unigram and BPE produced the best and similar results.

## Conformer-CTC Inference

Script for inferring data using Conformer-CTC:

```python
usage: transcribe_script.py [-h] --checkpoint_path CHECKPOINT_PATH --data_dir DATA_DIR [--output_csv OUTPUT_CSV] [--batch_size BATCH_SIZE]

Transcribe audio files and save to CSV.

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_path CHECKPOINT_PATH
                        Path to the .nemo ASR model checkpoint file.
  --data_dir DATA_DIR   Directory containing the .wav files.
  --output_csv OUTPUT_CSV
                        Output CSV file name.
  --batch_size BATCH_SIZE
                        Batch size for transcription.
```
## Datasets
checkpoints for the final results
https://www.kaggle.com/datasets/mohamedmotawie/final-submission
