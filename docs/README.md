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