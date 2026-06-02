## Introduction

This project builds a Text to Speech model for Hindi language. Here we are using [F5-TTS architecture](https://arxiv.org/abs/2410.06885) which is a fully non-autoregressive text to speech system. It is based on [Flow matching](https://arxiv.org/pdf/2210.02747) with Diffusion Transformer. We are using a pre-trained model that is trained on 100k+ hours of multi-lingual speech. In this project, we will be fine-tuning this model with nearly 600 hours of labelled Hindi language speech from IndicVoices dataset.

## Background
Text to speech systems are used to convert text to human like speech. It helps in making digital content more accessible and enhancing interactions between humans and machines. 

Text-to-speech (TTS) systems have evolved dramatically from rule-based synthesizers to modern neural network-based systems. State of the art models can now generate speech that is nearly indistinguishable from human recordings in terms of intelligibility and naturalness. 

### Early models
Since the 1980s and until about 2010, speech models were primarily stochasitc in nature. They combined 
1. hidden markov models: to model the sequence and duration of phonemes
2. gaussian mixture models: to model the association between acoustic features and phonemes

### Evolution
From stochastic models, TTS systems evolved to use modern neural pipelines typically consist of two stages: 
1. An acoustic model that converted text (or phoneme sequences) to a mel-spectrogram
2. A vocoder that converted the mel-spectrograms to a waveform

Acoustic modelling typically used Auto-regressive models with
1. phoneme sequencing 
2. phoneme durations modelling
3. mel-spectrograms generation

### State of art
The state of the art TTS models include
- Kokoro
- Fish Speech
- F5-TTS
- XTTS
- Dia
- Parler-TTS

These models use different neural network architectures and have MOS (Mean opinion score) of 4+, going upto 4.2. Where Human speech MOS score is 4.5+. 

In this project, we are going to focus on F5-TTS model architecture. It is a fully non-autoregressive text-to-speech system based on flow matching with Diffusion Transformer (DiT). Without requiring complex designs such as duration model, text encoder, and phoneme alignment, the text input is simply padded with filler tokens to the same length as input speech, and then the denoising is performed for speech generation. This approach was originally proved feasible by E2 TTS. The original design of E2 TTS makes it hard to follow due to its slow convergence and low robustness. To address these issues, F5-TTS first models the input with ConvNeXt to refine the text representation, making it easy to align with the speech. It further uses an inference-time Sway Sampling strategy, which significantly improves model's performance and efficiency. F5-TTS exhibits highly natural and expressive zero-shot ability, seamless code-switching capability, and speed control efficiency.

The F5-TTS architecture is open source and available under MIT license for further research and development.

## Problem Statement

The base F5-TTS model is trained on a public 100K hours dataset consisting mostly of english and chinese languages. It produces high quality human like speech for english texts. e.g.

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav, sr, spec = tts.infer(
    ref_file=str(files("data").joinpath("basic_ref_en.wav")),
    ref_text="some call me nature, others call me mother nature.",
    gen_text="I am a resident of the earth, and I am here to stay.",
    file_wave=str(files("data").joinpath("api_out_english.wav")),
    seed=None,
)
```
<audio controls><source src="working_output.wav" type="audio/wav"></audio>

However, when trying to use this model for Indian languages like Hindi, it produces garbled reference speech as it doesn't recognize Devnagiri script.

```python
wav, sr, spec = tts.infer(
    ref_file=str(files("data").joinpath("basic_ref_en.wav")),
    ref_text="some call me nature, others call me mother nature.",
    gen_text="मॆरा नाम आशिष है, और मैं पृथ्वी का निवासी हूँ।",
    file_wave=str(files("data").joinpath("api_out_hindi.wav")),
    seed=None,
)
```
<audio controls><source src="garbage_output.wav" type="audio/wav"></audio>

The objective of this project is to train the model with labelled Hindi language dataset. The dataset contains Hindi audio and transcriptions in Devnagiri script. Thus trained model can be used to convert Hindi Text provided in Devnagiri script to speech. 

## Dataset Description

To train the F5-TTS model for Hindi language, we are using [ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices#overview) dataset. It is a dataset of natural and spontaneous speech containing a total of 23.7K hours of read (8%), extempore (76%) and conversational (15%) audio from 51K speakers covering 400+ Indian districts and 22 languages. This work is funded by Bhashini, MeitY and Nilekani Philanthropies.

We will only be using Hindi language data samples from this dataset.

```python
from datasets import load_dataset
# Load the dataset from the HuggingFace Hub
dataset = load_dataset("ai4bharat/IndicVoices","hindi",split="train")
```

The transcriptions for the audio are provided in Devnagiri script in the dataset. We will be creating a vocabulary for TTS training from the transcriptions in Devnaigiri characters format.

## Model Architecture
F5-TTS model architecture is an improvement on the E2-TTS architecture. Key components of the E2-TTS architecture are 
1. The backbone of the model is a Transformer architecture. 
2. It incorporates U-Net style skip connections.
3. It consists of 24 layers, 16 attention heads, an embedding dimension of 1024, a linear layer dimension of 4096. 
4. The character embedding vocabulary size is 399.
5. The total number of parameters amounts to 335 million. 
6. The input is modelled as 
    - 100-dimensional log mel-filterbank features
    - extracted every 10.7 milliseconds from audio samples 
    - at 24 kHz sampling rate 
7. A BigVGAN based vocoder is employed to convert the log mel-filterbank features into waveforms.

![E2-TTS Architecture](E2-TTS-Arch.png)

8. F5-TTS architecture adds onto the E2-TTS with the use of ConvNext V2 to refine the text representation.

![F5-TTS Architecture](F5-TTS-Arch.png)

## Theoretical Analysis

### Training steps for each sample
1. The transcription of training sample $y = (c_1, c_2, ..... c_m)$ is extended with filler tokens $\to y^* = (c_1, c_2, ..... c_m, <F>,...<F>)$
2. $s^*$ mel-filterbank features are extracted from the training audio
3. A span of the mel-spectrogram is masked out. This is the infilling task
4. Model is trained to learn $P(m.s^* | (1-m).s^*,y^*)$
5. m is a mask matrix of random length
6. The learned distribution is used to predict the mel-filterbanks during inference

### Inference steps for each sample
1. An audio prompt $s^{aud}$ is provided as a reference for generating similar sounding speech. 
2. Reference transcription $y^{aud}$ of $s^{aud}$ is provided 
3. A text prompt $y^{gen}$ is provided to generate speech
4. A target duration is set $T^{gen}$
5. $s^{*aud}$ mel-filterbank features are extracted from $s^{aud}$
6. $y^* = (y^{aud}, y^{gen}, <F>...\tau times)$ is generated by concatenating reference transcription with input text. This is extended with filler tokens
7. The model outputs $s^*$ such that $P(s^*|[s^{aud};m^{gen}],y^*)$ is maximized
8. $s^*$ is converted to audio waveforms by the vocoder

## Data preprocessing
For finetuning the F5-TTS model with IndicVoices dataset, we performed following preprocessing steps.

1. Use the audio waveform and transcriptions from the IndicVoices dataset
```python
# Keep only audio_filepath and text columns
dataset = dataset.select_columns(["audio_filepath", "text"])
```

2. Parse all the transcriptions to identify the vocabulary for training
```python
# Go through the text to identify all the graphemes in the dataset
graphemes = set()
for text in tqdm(dataset["text"]):
    for char in text:
        graphemes.add(char)

graphemes = sorted(list(graphemes))
```

3. Parse all audio waveforms and save to .wav files in a folder
```python
import soundfile as sf

records = []
for i in tqdm(range(max_samples), desc="Extracting audio"):
    audio = dataset[i]["audio_filepath"]
    audio_array = audio["array"]
    sample_rate = audio["sampling_rate"]
    text = dataset[i]["text"]
    
    # Save audio file
    wav_path = wavs_dir / f"hindi_{i:05d}.wav"
    sf.write(wav_path, audio_array, sample_rate)
    
    records.append({
        "audio_file": str(wav_path),
        "text": text
    })
```

4. Save the mapping of the .wav files and transcriptions in a csv file
```python
df = pd.DataFrame(records)
df.to_csv("metadata.csv", sep="|", index=False, header=True)
```

5. The F5-TTS training is performed using metadata.csv. The metadata file contains absolute paths for the .wav files for audio.  

## Training Methodology

F5-TTS model provides a [training cum fine-tuning utility](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/train) for ease of use. We made use of this utility to train the model with following configuration

```yaml
datasets:
  name: Hindi  # dataset name
  batch_size_per_gpu: 2  # 8 GPUs, 8 * 38400 = 307200
  batch_size_type: sample  # frame | sample
  max_samples: 32  # max sequences per batch if use frame-wise batch_size. we set 32 for small models, 64 for base models
  num_workers: 6

optim:
  epochs: 1
  learning_rate: 7.5e-5
  num_warmup_updates: 200  # warmup updates
  grad_accumulation_steps: 1  # note: updates = steps / grad_accumulation_steps
  max_grad_norm: 1.0  # gradient clipping
  bnb_optimizer: False  # use bnb 8bit AdamW optimizer or not

model:
  name: F5TTS_Hindi  # model name
  tokenizer: custom  # tokenizer type
  tokenizer_path: data/Hindi_custom/vocab.txt  # if 'custom' tokenizer, define the path want to use (should be vocab.txt)
  backbone: DiT
  arch:
    dim: 1024
    depth: 18
    heads: 12
    ff_mult: 2
    text_dim: 512
    text_mask_padding: True
    qk_norm: null  # null | rms_norm
    conv_layers: 4
    pe_attn_head: null
    attn_backend: torch  # torch | flash_attn
    attn_mask_enabled: False
    checkpoint_activations: False  # recompute activations and save memory for extra compute
  mel_spec:
    target_sample_rate: 24000
    n_mel_channels: 100
    hop_length: 256
    win_length: 1024
    n_fft: 1024
    mel_spec_type: vocos  # vocos | bigvgan
  vocoder:
    is_local: False  # use local offline ckpt or not
    local_path: null  # local vocoder path
```

The training is being run for 100 epochs.

## Experimental Results
After training the model with a small dataset of 10k samples and 1 epoch, it is still producing garbled output. However, it is not just using the reference speech anymore. This implies, it has started to learn the new vocabulary

```python
tts_hindi = F5TTS(model="F5TTS_Hindi", ckpt_file=fine_tuned_tts_model_ckpt, vocab_file=vocab_file_path)
wav, sr, spec = tts_hindi.infer(
    ref_file=str(files("data").joinpath("basic_ref_en.wav")),
    ref_text="some call me nature, others call me mother nature.",
    gen_text="मॆरा नाम आशिष है, और मैं पृथ्वी का निवासी हूँ।",
    file_wave=str(files("data").joinpath("api_out_hindi.wav")),
    file_spec=str(files("data").joinpath("api_out_hindi.png")),
    seed=None,
)
```
<audio controls><source src="noisy_output.wav" type="audio/wav"></audio>

Due to scarcity of compute resources, the training is still going-on. Final results are still a work in-progress.

## Insights
There are multiple benefits of F5-TTS architecture over alternatives
1. **Improved inference latency**: It is a non-regressive model. This results in low inference latency due to the ability of the model to parallely generate the mel-filterbanks
2. **Minimal data preprocessing**: It models a infilling task to directly work with character sequence followed by filler tokens. So, it doesn't need to perform grapheme to phoneme conversion. This results in minimal data preprocessing requirement, as it generates the mel-filterbanks directly based on the input sequence of characters in the written script of the language.
3. **Models multiple tasks**: Effectively, it uses a single neural network to model 
    - Grapheme to phoneme converter
    - Phoneme duration model
    - Mel-spectrogram generator
4. **Low complexity model**: As the number of model parameters is dependent on the vocabulary size, we can keep the model small by building dedicatedly for a single language or a family of related languages. e.g. Hindi, Marathi, Nepali, Konkani share the same Devnagiri script. A single model can be used as TTS system for these languages with a small common vocabulary.

## Glossary
- **TTS**: It stands for Text to Speech. It is a computer program that converts written text in a given language to Human like audio speech.
- **Grapheme**: It is a letter or letters that represent a sound. It can be a single letter or a combination of letters. In hindi language, each character is a grapheme.
- **Phoneme**: It is a small meaningful unit of sound. There can be many to many mapping between graphemes and phonemes as multiple letters can have the same sound andvice versa.
- **Auto Regressive model**: A model that predicts the next element in a sequence by conditioning its output on all previously generated tokens. This sequential dependency ensures high logical coherence and "flow," but it makes inference relatively slow since each step must wait for the preceding one to finish.
- **Non Auto regressive model**: Unlike their sequential counterparts, non-auto-regressive models generate all elements of a sequence simultaneously or in parallel. This approach significantly boosts inference speed.
- **Flow Matching**: Flow matching is a training framework for Continuous Normalizing Flows that simplifies the learning of probability paths by regressing a vector field directly against a target velocity. By matching a vector field to a predefined conditional probability path, it bypasses the need for expensive ODE solvers during training, often resulting in faster sampling and more stable optimization than traditional diffusion models.
- **Diffusion Transformer**: It is a generative neural network architecture that replaces the standard U-Net backbone of a diffusion model with a Transformer-based structure. By applying Transformers to latent noise, DiTs leverage the scalability and global attention mechanisms of the Transformer to produce state-of-the-art results in audio generation.
- **Mel-spectrogram**: It is a time-frequency representation of audio where the frequencies are converted to the Mel scale, which mimics the non-linear way human ears perceive pitch. It is created by passing a standard spectrogram through a Mel-filterbank, making it a more efficient and perceptually relevant feature for training speech models.
- **Mel-filterbank**: It is a collection of overlapping triangular filters used to transform a linear frequency power spectrum into the Mel scale. These filters are spaced closer together at lower frequencies and farther apart at higher frequencies to capture the nuances of human hearing and reduce the dimensionality of audio data.
- **Vocoder**: It is a neural network (like HiFi-GAN or WaveNet) that converts acoustic features, such as Mel-spectrograms, back into audible, raw waveforms. Its primary job is to reconstruct the phase information lost during the spectrogram creation process to ensure the resulting audio sounds natural and fluid.
- **ConvNext**: ConvNext is a "modernized" convolutional neural network (CNN) architecture that adopts design choices from Vision Transformers (ViTs), such as larger kernel sizes (7x7) and inverted bottlenecks.

## References
- [E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS](https://arxiv.org/abs/2406.18009)
- [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://arxiv.org/abs/2410.06885)