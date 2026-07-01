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
- Count number of parameters in base model: 337 million
- To run one iteration of training on base model and check the loss. This will give rough idea on how much loss function value do we need to reach to generate a good output. 
- The BnB optimizer helped in initial training by reducing memory usage. Now try switching it off so that quantization losses can be minimized. 
- Find distribution of occurrence of each character in vocab. 
- Find distribution of occurrence of each character and vowel combination in vocab. 
- Find distribution of occurrence of each word in vocab. Since there would be too many words, calculate mean and variance of all words.
- Find the distribution of duration of samples.
- Try sorting the samples based on duration. This would play similar duration samples together. Each batch used for training will have similar durations. This may reduce the number of dummy computations due to padding. 
- Check input dimension and convNext modeling. Is convNext helping with Hindi speech? 

## Jarvis labs GPU usage
- To install jarvislabs cli
```pip install jarvislabs```
- To complete project setup
```jl setup```
- To check account status
```jl status```