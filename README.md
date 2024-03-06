# RMVPE

This repo is the Pytorch implementation of ["RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music"](https://arxiv.org/abs/2306.15412v2). 

---
Update by yamathcy
- Add single file inference
  - Usage 
  ```angular2html
    python single_file_test.py --file_path {audio file to be estimated} --model_path {your model path} --device {your device}
  ```
  This model checkpoint can be used for inference: https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt