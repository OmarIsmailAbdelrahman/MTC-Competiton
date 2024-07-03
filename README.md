# MTC-Competition

## Experiments

### 1. [Wav2Vec2][1]
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

### 2. FAdam optimizer [2]

introduces several improvements over traditional Adam, particularly benefiting Automatic Speech Recognition (ASR):
#### Key Improvements
1. **Enhanced Performance in ASR**:
    - **State-of-the-Art Results**: Coupled with w2v-bert, achieves new state-of-the-art word error rates (WER) on the LibriSpeech dataset with a 600 million parameter Conformer model, outperforming Adam.
    - **Superior Fine-Tuning**: Demonstrates significant improvements in semi-supervised fine-tuning of ASR models, leading to better overall accuracy and reliability.
2. **Adaptive Mechanisms**: Includes adaptive epsilon, gradient clipping, and refined weight decay, which enhance stability and reduce the need for extensive hyperparameter tuning.

3. **Robust Convergence**: Provides stronger convergence guarantees through the application of momentum to the natural gradient, making the optimization process more reliable.

### 3. Knowledge Distillation Model
- Attempted knowledge distillation with Wav2Vec2 and Conformer-CTC but failed due to mismatched decoder lengths.
- Potential solution: Adjust transformer heads to match the output.

## Final approach: Conformer CTC [3]  using NeMo Framework
Currently, conformer and Fast conformer[4] based models hold the state-of-the-art performance for speech recognition and processing tasks.
The Conformer-CTC model is a type of neural network architecture that combines the strengths of convolutional neural networks (CNNs),
transformers, and CTC for speech recognition tasks.

<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/81030289/455cf6ab-9138-478a-b63a-ff8617909da4" alt="Conformer architecure"/>
</p>

### Configurations and trials:
- Utilized FAdam for optimization.
-  Training with precision 32 and 16
#### Tokenizers Configurations:
  - Added custom \<fill> \<overlap> \<laugh> tokens
  - Tokenization types:
    - **wpe:** Handles out-of-vocabulary words by breaking them into subwords, but requires a well-prepared corpus for effective training.
    - **SPE:** suitable for scripts with or without clear word boundaries, handles dialects like Egyptian Arabic well, 
    but quality depends heavily on the training data's quality and size.
  - Tokenization techniques:
     - **Unigram**: Utilizes a subword segmentation algorithm based on a unigram language model. It aims to find the most likely subword units given the training data, maximizing the likelihood of the training corpus.
     - **BPE (Byte-Pair Encoding)**: Iteratively merges the most frequent pairs of bytes or characters. It is particularly effective for text compression and is widely used in neural machine translation systems. BPE is the default tokenization type in SentencePiece.
     - **Char**: Treats each character as a token.
    

There are two losses/decoders that can be used with a conformer, conformer-ctc and conformer-transducer.
We decided to use conformer-ctc as its less computationally expensive and yields good results.

| Feature             | Conformer-CTC                         | Conformer-Transducer                    |
|---------------------|---------------------------------------|-----------------------------------------|
| **Architecture**    | Conformer layers + CTC loss           | Conformer encoder + transducer framework|
| **Loss Function**   | Connectionist Temporal Classification (CTC) | Transducer (RNN-T)                      |
| **Output**          | Probability distribution over sequences | Sequence of probability distributions conditioned on input and previous outputs |
| **Advantages**      | Simpler, effective for alignment tasks, less computationally intensive | Handles long-range dependencies, better for complex alignments, suitable for real-time applications |
| **Disadvantages**   | Struggles with long sequences, less effective with context | More complex, computationally intensive |

#### Spec Augment:
SpecAugment is a data augmentation technique specifically designed for speech recognition tasks. It applies various transformations to the spectrograms of audio signals to make the model more robust and improve its generalization capability. Here are the key components of SpecAugment:

1. **Time Warping**:
   - Time warping shifts the spectrogram in the time direction. This simulates slight variations in speaking speed and timing, making the model more robust to variations in speech patterns.

2. **Frequency Masking**:
   - In frequency masking, one or more ranges of frequencies are masked (set to zero) randomly. This means certain frequency bands are removed, which forces the model to learn to handle missing frequency information. This technique helps the model become more invariant to different acoustic conditions.

3. **Time Masking**:
   - Similar to frequency masking, time masking involves masking out one or more time segments of the spectrogram. This means certain time intervals are set to zero, encouraging the model to handle parts of the audio being missing or occluded, which can occur in real-world scenarios like overlapping speech or background noise.

These augmentations are applied randomly and independently during training, which helps create a more diverse training set and prevents overfitting.




#### Challenges faced during training:
- The initial nemo config for the model used a very high learning rate which made
using bpe and char tokenizers based models show no predictions even after training for many epochs but worked fine for
unigram based tokenizers which was confusing.


### 3. KenLM N-gram
- Created multiple n-grams (3 to 6) using the train/adapt dataset, resulting in small output changes.
- Constructed n-grams using the Egyptian Datasets Collection (2.5 million rows), improving performance slightly.
- We tried to integrate it with the acoustic model, but the results was worse or no difference :(


### final model configuration:
- Conformer-CTC Large
- precision: 32
- tokenizer: spe unigram
- optimizer: Fadam

### result
 - This one of the results of the unigram tokenizer model. 
This due to the high learning rate set by default, which we discovered late.
<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/73082049/2013718c-4a09-49d1-af5e-d3dd6ec0f8bb" alt="Conformer CTC on NeMo"/>
</p>

- After reducing the learning rate and using char tokenizer, a smooth decrease in wer was achieved but this model needed more training and we did not have enough time to submit its results.
<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/81030289/9af5b2d8-e3b4-4ad3-b152-12679b4a36d4" alt="Char tokenizer results"/>
</p>

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



## Checkpoints
The checkpoint is in this [kaggle dataset](https://www.kaggle.com/datasets/bigsus/final-submission
)
```
/final-submission/results/Some name of our experiment/checkpoints/Some name of our experiment.nemo
```

## references
[1]: https://arxiv.org/pdf/2006.11477 <br>
[2] https://arxiv.org/pdf/2405.12807v7 <br>
[3] https://arxiv.org/pdf/2005.08100 <br>
[4] https://arxiv.org/pdf/2305.05084 <br>