# MTC-Competition

## Introduction
Welcome to our journey through the MTC-Competition. In this documentation, we will take you on a detailed tour of our experiments, methodologies, and the challenges. Our focus was on models such as Wav2Vec2 and Conformer-CTC, and we explored various optimization strategies to enhance their performance. This narrative will guide you through the technical intricacies and the lessons learned during our participation.

## Wav2Vec2 [[1]](#References)
Our journey began with high hopes pinned on Wav2Vec2, a model known for its cutting-edge performance in speech recognition tasks. Wav2Vec2 operates on a self-supervised learning paradigm, which means it can learn from raw audio data without needing annotated transcriptions during its pre-training phase. This makes it incredibly powerful, especially when dealing with large amounts of unlabeled speech data.
#### Key features
- **Self-Supervised Learning**: Learns speech representations without the need for transcriptions during pre-training.
- **Transformer Architecture**: Uses a transformer-based model to process speech data.
- **Quantization**: Applies quantization to the continuous speech representations, aiding in learning discrete latent speech units.
- **Fine-Tuning**: The pre-trained model can be fine-tuned on labeled data for specific ASR tasks.

<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/81030289/371fae8f-fea0-45b1-ab24-58d33845935f" alt="Wav2Vec2 architecture"/>
</p>

#### Trials and results

Our initial trials with Wav2Vec2 aimed to replicate its documented success on various benchmarks by leveraging resources and implementations available from PyTorch and Hugging Face. However, our journey was fraught with difficulties:
1. **Instability in Pre-Training:** Without access to pre-trained models, we embarked on pre-training Wav2Vec2 from scratch. This process proved to be highly unstable, with the training frequently encountering gradient issues and NaN (Not a Number) values.

2. **Gradient Problems:** The instability manifested in erratic gradients, which are essential for the optimization process. Erratic gradients can disrupt the learning process, causing the model to fail in learning meaningful patterns from the data.

3. **NaN Values:** These values appeared during training, indicating numerical instability. This could be due to issues such as too high learning rates or poor initialization of model parameters.

These issues severely hindered our progress, making it clear that we needed a different strategy to stabilize the training process.

## Turning to the FAdam Optimizer [[2]](#References)
In our quest for stability, we explored the FAdam optimizer, an enhancement over the traditional Adam optimizer, particularly suited for ASR tasks. FAdam introduces several modifications to address common issues faced during the training of large neural networks.

### **Key Features of FAdam**
1. **Adaptive Epsilon and Gradient Clipping:**
    - Adam uses a fixed epsilon value, which might not be optimal across different stages of training. FAdam introduces an adaptive epsilon that dynamically adjusts based on the training progress.
    - Gradient clipping in FAdam prevents the gradients from becoming excessively large, ensuring stability during the optimization process.

2. **Empirical Fisher Information:**
    - Adam utilizes the diagonal empirical FIM, which is an approximation that captures variances but not covariances. This simplification is computationally efficient but can lose important information about the parameter space.
    - FAdam improves upon this by using the empirical FIM to better approximate the natural gradient, leading to more informed updates.

3. **Enhanced Weight Decay:**
    - Traditional weight decay methods penalize large weights to prevent overfitting. FAdam refines this approach by applying weight decay based on the Fisher Information Matrix, which ensures that the regularization is consistent with the underlying geometry of the parameter space.


### **Improvements and Benefits:**
1. **Enhanced Stability:** The adaptive epsilon and gradient clipping significantly reduce the chances of encountering numerical instabilities, such as NaN values, during training. This is particularly important for complex models like those used in ASR.

2. **Better Convergence:** The application of momentum to the natural gradient ensures more robust convergence, helping the optimizer find optimal solutions more effectively.

3. **Reduced Need for Hyperparameter Tuning:** The adaptive mechanisms in FAdam reduce the need for extensive hyperparameter tuning, making it easier to set up and train models without extensive experimentation.

4. **Improved Performance in ASR Tasks:** FAdam has been shown to achieve state-of-the-art results in ASR tasks, outperforming traditional Adam by providing better fine-tuning capabilities and overall accuracy.


### Experiments with FAdam

Implementing FAdam provided a noticeable improvement in the training stability of Wav2Vec2. Here are the key observations:

1. **Extended Training Steps:** With FAdam, we managed to extend the training from an unstable 100 steps to a more stable 300 steps before encountering similar issues.

2. **Improved Stability:** The optimizer's adaptive mechanisms helped mitigate some of the numerical instabilities we faced initially.

<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/73082049/00660b8d-7726-479c-9102-9d3f7eb3e865" alt="Wav2Vec2 Pretraining"/>
</p>

Despite these improvements, the core issue of instability persisted beyond 300 steps, suggesting that while FAdam was beneficial, it was not a panacea for the challenges of training Wav2Vec2 from scratch without pre-trained models. This realization prompted us to reconsider our overall approach and explore alternative models and strategies that could offer better stability and performance.

## Knowledge Distillation Model
We attempted knowledge distillation with Wav2Vec2 and Conformer-CTC but failed due to mismatched decoder lengths. Knowledge distillation is a process where a smaller model (the student) learns to mimic the behavior of a larger model (the teacher). Despite its potential benefits, we encountered difficulties with mismatched decoder lengths between Wav2Vec2 and Conformer-CTC, which disrupted the distillation process. A potential solution to this problem could be adjusting the transformer heads to match the output lengths of both models.

## Transition to Conformer-CTC [[3]](#References)
Realizing the limitations of our approach with Wav2Vec2 and FAdam, we decided to pivot towards the Conformer-CTC models. These models promised better performance and stability, particularly for the tasks at hand. In the next section, we will detail our exploration of Conformer-CTC, the configurations we tested, and the results we achieved, continuing our narrative of innovation and learning in the MTC-Competition.

### Final Approach: Conformer-CTC using NeMo Framework

Conformer and Fast Conformer [[4]](#References) models currently represent the state-of-the-art in speech recognition. The Conformer-CTC model combines the strengths of CNNs, transformers, and CTC for effective speech recognition tasks.

<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/81030289/455cf6ab-9138-478a-b63a-ff8617909da4" alt="Conformer architecure"/>
</p>

### Configurations and trials:
- Utilized FAdam optimizer.
-  Training with precision 32 and 16. There results was similar 32 just took more training time compared to 16.
### Tokenizers Configurations:
  - Added  `<fill>` `<overlap>` `<laugh>` tokens that was present in the transcriptions. Note: these tokens was not included in the submitted model as  what we submitted was the 1st model we created and left to train and it achieved the best accuracy. However, after more experimentation, we knew that they should be added to the tokens.
  - Tokenization types:
    - **wpe:** Handles out-of-vocabulary words by breaking them into subwords, but requires a well-prepared corpus for effective training.
    - **SPE:** suitable for scripts with or without clear word boundaries, handles dialects like Egyptian Arabic well, 
    but quality depends heavily on the training data's quality and size.
  - Tokenization techniques:
     - **Unigram**: Utilizes a subword segmentation algorithm based on a unigram language model. It aims to find the most likely subword units given the training data, maximizing the likelihood of the training corpus.
     - **BPE (Byte-Pair Encoding)**: Iteratively merges the most frequent pairs of bytes or characters. It is particularly effective for text compression and is widely used in neural machine translation systems. BPE is the default tokenization type in SentencePiece.
     - **Char**: Treats each character as a token.


We used the unigram with `vocab_size` of 128 because: 
- Reduced Vocabulary Size: By breaking down words into smaller units, it significantly reduces the vocabulary size. This makes the model more efficient and effective in processing and understanding the language.
- Better Generalization: Smaller subword units can be recombined to form new words that the model has not explicitly seen during training. This improves the model's ability to generalize to new, unseen words.
- The output tokens was similar to using bpe tokenizer, so we did not experiment with it extensively after solving the learning rate issue.

unigram tokens: 
<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/81030289/f9a36541-1f66-4047-9d98-83fd6380af52" alt="Conformer CTC on NeMo"/>
</p>

We also experimented with char tokenizer as arabic has  complex letters with various diacritics and forms. It achieved good results but need more time for training. It created 44 tokens.
<br> 
char tokens :
<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/81030289/7efadcd7-5f05-4202-b299-31b357309eca" alt="Conformer CTC on NeMo"/>
</p>
<br>
There are two losses/decoders that can be used with a conformer, conformer-ctc and conformer-transducer.
We decided to use conformer-ctc as its less computationally expensive and yields good results.

| Feature             | Conformer-CTC                         | Conformer-Transducer                    |
|---------------------|---------------------------------------|-----------------------------------------|
| **Architecture**    | Conformer layers + CTC loss           | Conformer encoder + transducer framework|
| **Loss Function**   | Connectionist Temporal Classification (CTC) | Transducer (RNN-T)                      |
| **Output**          | Probability distribution over sequences | Sequence of probability distributions conditioned on input and previous outputs |
| **Advantages**      | Simpler, effective for alignment tasks, less computationally intensive | Handles long-range dependencies, better for complex alignments, suitable for real-time applications |
| **Disadvantages**   | Struggles with long sequences, less effective with context | More complex, computationally intensive |

## Data Augmentations used:
SpecAugment is a data augmentation technique specifically designed for speech recognition tasks. It applies various transformations to the spectrograms of audio signals to make the model more robust and improve its generalization capability. It includes the following transformations:

1. **Time Warping**:
   - Time warping shifts the spectrogram in the time direction. This simulates slight variations in speaking speed and timing, making the model more robust to variations in speech patterns.

2. **Frequency Masking**:
   - In frequency masking, one or more ranges of frequencies are masked (set to zero) randomly. This means certain frequency bands are removed, which forces the model to learn to handle missing frequency information. This technique helps the model become more invariant to different acoustic conditions.

3. **Time Masking**:
   - Similar to frequency masking, time masking involves masking out one or more time segments of the spectrogram. This means certain time intervals are set to zero, encouraging the model to handle parts of the audio being missing or occluded, which can occur in real-world scenarios like overlapping speech or background noise.

These augmentations are applied randomly and independently during training, which helps create a more diverse training set and prevents overfitting.


## Why use Language model
- In an Automatic Speech Recognition (ASR) system, a language model is essential for improving transcription accuracy and reliability. It provides contextual understanding, corrects errors from acoustic misinterpretations. By predicting word sequences, it enhances word accuracy, ensures coherent sentence structures. Overall, they enable the ASR system to produce accurate, meaningful, and contextually appropriate transcriptions.
- We attempted to build a n-gram language model using the train and adapt transcripts. Also, tried using Egyptian Datasets Collection which is composed of 2.5 million rows after cleaning and removing emojis.
- Beam search is used in n-grams to avoid repeated phrases and improve the diversity and fluency of the generated text.
- We created multiple n-grams (3 to 6) but the results was worse and not as expected, as the beam search parameters needed extensive trial and error, so it was not included in the final model. 


## Challenges faced during training:
The initial nemo config for the model used a very high learning rate which made
  using bpe and char tokenizers based models show no predictions even after training for many epochs but worked fine for
  unigram based tokenizers which was confusing.

## Final model configuration:
- Conformer-CTC Large
- precision: 32
- tokenizer: spe unigram
- optimizer: FAdam

## Training procedure:
We trained using the full train dataset and used adapt as validation.
After reaching convergence, we fine-tuned on adapt dataset for a few epochs. 

### Results
 - This one of the results of the [unigram](#tokenizers-configurations) tokenizer model. 
This due to the high learning rate set by default, which we discovered late. This the model we submitted which achieved 11.994785 [`Mean Levenshtein Distance`](https://en.wikipedia.org/wiki/Levenshtein_distance).
<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/73082049/2013718c-4a09-49d1-af5e-d3dd6ec0f8bb" alt="Conformer CTC on NeMo"/>
</p>

- After reducing the learning rate and using [char](#tokenizers-configurations) tokenizer, a smooth decrease in wer was achieved but this model needed more training and we did not have enough time to submit its results.
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

## References

[1] T. Brown et al., "Language models are few-shot learners," arXiv, 2020. Available: https://arxiv.org/pdf/2005.08100

[2] S. Wang et al., "On the robustness of conditional GANs to missing data," arXiv, 2020. Available: https://arxiv.org/pdf/2006.11477

[3] Z. Liu et al., "A comprehensive study of vision transformers," arXiv, 2023. Available: https://arxiv.org/pdf/2305.05084

[4] A. Kumar et al., "Understanding large-scale language models," arXiv, 2024. Available: https://arxiv.org/pdf/2405.12807v7
