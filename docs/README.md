# How to use this repo for training F5TTS model for Hindi

## Data preparation
- Data preparation for Hindi language script can be found at src/f5_tts/train/datasets/prepare_hindi.py
- The script can be executed from cli as
```
python3 src/f5_tts/train/datasets/prepare_hindi.py {hugging_face_token} ./data/hindi_custom
```

## Training
- The training configuration file can be found at src/f5_tts/configs/F5TTS_Hindi_deeper.yaml
- The training script can be executed from cli as
```
python3 src/f5_tts/train/train.py --config-name F5TTS_Hindi_deeper.yaml
```

## Evaluation
- To evaluate the model checkpoint use the script at src/f5_tts/eval/eval_hindi.py
- Execute evaluation script from cli as
```
python3 src/f5_tts/eval/eval_hindi.py --config-name F5TTS_Hindi_deeper --ckpt ckpts/F5TTS_Hindi_deeper_vocos_custom_hindi/model_last.pt --vocab data/hindi_custom/vocab.txt --ref-audio-file audio.wav --output-file output.wav
```

```
python3 src/f5_tts/eval/eval_base.py --config-name F5TTS_v1_Base --vocab data/F5TTS_v1_Base/vocab.txt --ref-audio-file english_ref_audio.wav --output-file english_output.wav
```


## To do
- Count number of parameters in base model: 
    - 337 million
- To run one iteration of training on base model and check the loss. This will give rough idea on how much loss function value do we need to reach to generate a good output.
    - The loss function value is >1 when trying to finetune the base model on Emilia dataset. 
- The BnB optimizer helped in initial training by reducing memory usage. Now try switching it off so that quantization losses can be minimized. 
    - There doesn't seem to be any significant difference in loss function value when BnB optimizer is used or not
- Find distribution of occurrence of each character in vocab. 
- Find distribution of occurrence of each character and vowel combination in vocab. 
- Find distribution of occurrence of each word in vocab. Since there would be too many words, calculate mean and variance of all words.
- Find the distribution of duration of samples.
- Try sorting the samples based on duration. This would play similar duration samples together. Each batch used for training will have similar durations. This may reduce the number of dummy computations due to padding. 
- Check input dimension and convNext modeling. Is convNext helping with Hindi speech? 
- Filtering bad alignments
    - What "alignment" means here: the correspondence between the transcript and the audio — does the text actually match what is spoken, over the right time span? A "bad alignment" is a sample where text and audio disagree. F5-TTS learns text→speech, so a mismatched pair teaches it the wrong thing and shows up as instability or hallucination at inference. e.g. Extra audio — leading/trailing silence, music, or a second speaker not in the transcript. Speed mismatch — text implies far more or fewer characters than the audio duration supports (e.g. 8 seconds of audio with 3 characters, or 1 second with 200 characters). Empty / near-empty text or audio.
    - Duration bounds. Drop clips shorter than ~0.5 s or longer than 95% (~20 s).
    - Characters-per-second sanity. Compute len(normalized_text) / audio_duration. Determine heuristics and apply filter. Points far outside the band are almost always misaligned — drop them. This one heuristic catches a surprising amount of garbage.
    - Silence/energy check. Drop clips whose RMS energy is near zero, or trim leading/trailing silence (VAD) so the audio span matches the transcript span.

## Jarvis labs GPU usage
- To install jarvislabs cli
```pip install jarvislabs```
- To complete project setup
```jl setup```
- To check account status
```jl status```