# MTC-Competition

## Experiments

### 1. Wav2Vec2 [1]
#### Key features
- **Self-Supervised Learning**: Learns speech representations without the need for transcriptions during pre-training.
- **Transformer Architecture**: Uses a transformer-based model to process speech data.
- **Quantization**: Applies quantization to the continuous speech representations, aiding in learning discrete latent speech units.
- **Fine-Tuning**: The pre-trained model can be fine-tuned on labeled data for specific ASR tasks.

<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/81030289/371fae8f-fea0-45b1-ab24-58d33845935f" alt="Wav2Vec2 architecture"/>
</p>

#### Trials and results
We aimed to use wav2vec2 as it had achieved state-of-the-art on multiple datasets 
and had implementations and documentations from popular libraries and sources like pytorch and huggingface.
However, due to the limitations of not using pre-trained models, we faced issues and instability in 
the pre-training stage of wav2vec2.
- Encountered instability, leading to issues with the gradient and NaN values.
- Instability occurred after 100 steps with FAdam and 300 steps with WAdam.

<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/73082049/00660b8d-7726-479c-9102-9d3f7eb3e865" alt="Wav2Vec2 Pretraining"/>
</p>

### 2. FAdam optimizer

introduces several improvements over traditional Adam, particularly benefiting Automatic Speech Recognition (ASR):
#### Key Improvements
2. **Enhanced Performance in ASR**:
    - **State-of-the-Art Results**: Coupled with w2v-bert, achieves new state-of-the-art word error rates (WER) on the LibriSpeech dataset with a 600 million parameter Conformer model, outperforming Adam.
    - **Superior Fine-Tuning**: Demonstrates significant improvements in semi-supervised fine-tuning of ASR models, leading to better overall accuracy and reliability.
3. **Adaptive Mechanisms**: Includes adaptive epsilon, gradient clipping, and refined weight decay, which enhance stability and reduce the need for extensive hyperparameter tuning.

4. **Robust Convergence**: Provides stronger convergence guarantees through the application of momentum to the natural gradient, making the optimization process more reliable.

### 3. Knowledge Distillation Model
- Attempted knowledge distillation with Wav2Vec2 and Conformer-CTC but failed due to mismatched decoder lengths.
- Potential solution: Adjust transformer heads to match the output.

### 4. Conformer CTC on NeMo
- Achieved the best and most stable results compared to Wav2Vec2.
- Overfitted in the normal configuration at 25 epochs; fine-tuned for an additional 10 epochs.
- Utilized FAdam for optimization.

<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/81030289/455cf6ab-9138-478a-b63a-ff8617909da4" alt="Conformer architecure"/>
</p>
<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/73082049/2013718c-4a09-49d1-af5e-d3dd6ec0f8bb" alt="Conformer CTC on NeMo"/>
</p>


### 3. KenLM N-gram
- Created multiple n-grams (3 to 6) using the train/adapt dataset, resulting in small output changes.
- Constructed n-grams using the Egyptian Datasets Collection (2.5 million rows), improving performance slightly.

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

## references
[1] https://arxiv.org/pdf/2006.11477
[2] https://arxiv.org/pdf/2405.12807v7