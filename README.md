# MTC-Competiton
 ASR Competition
1. Conformer CTC on NeMo
Achieved the best and most stable results compared to Wav2Vec2.
Overfitted in the normal configuration at 25 epochs; fine-tuned for an additional 10 epochs.
Utilized FAdam for optimization.
2. Wav2Vec2 Pretraining
Pretrained Wav2Vec2 but encountered instability, leading to NaN values.

Instability occurred after 100 epochs with FAdam and 300 epochs with WAdam.
![image](https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/73082049/cfc88c19-b8f6-489c-b6d0-59e256914e8f)

3. KenLM N-gram
Created multiple n-grams (3 to 6) using the train/adapt dataset, resulting in small output changes.
Constructed n-grams using the Egyptian Datasets Collection (2.5 million rows), improving performance slightly from 13% to 11% WER.
4. Knowledge Distillation Model
Attempted knowledge distillation with Wav2Vec2 and NeMo but failed due to mismatched decoder lengths.
Potential solution: Adjust transformer heads to match the output.
5. FAdam Optimization
Used FAdam to reduce training time and enhance model stability.

Demonstrated better results with NeMo CTC Conformer, reaching 35 epochs before overfitting, and 25 epochs in normal training.
![image](https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/73082049/00660b8d-7726-479c-9102-9d3f7eb3e865)

6. Tokenizer Experiments
Tried different tokenizers; unigram and BPE produced the best and similar results.
