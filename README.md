# Robin3D

This is an official repo for paper "Robin3D: Improving 3D Large Language Model via Robust Instruction Tuning", ICCV 2025.
[[paper](https://arxiv.org/pdf/2410.00255)]


## 🔨 Preparation

- Prepare the environment:
  
  ```shell
  conda create -n chat-3d-v2 python=3.9.17
  conda activate chat-3d-v2
  conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
  pip install -r requirements.txt
  ```
  
- Download LLM backbone:
  -  We use Vicuna-7B v1.5 in our experiments, which can be downloaded from [Hugging Face](https://huggingface.co/lmsys/vicuna-7b-v1.5).

  - Change the `llama_model_path` in [config.py](./scripts/config.py) to the location of `vicuna-7b-v1.5`.
  

- Annotations and extracted features:

  You can download them from [Google Drive](https://drive.google.com/drive/folders/14Si8bdWI3N5NEeVDLhmAlxilWPl0f_Wp?usp=sharing). Please check the ```scripts/config.py``` for better understanding of the files on Google Drive.
  
  The detailed instructions are in [Chat-Scene's Preparation](https://github.com/ZzZZCHS/Chat-Scene/tree/dev/preprocess).


## 🤖 Training and Inference

- Training
  - Stage1:
    ```
    bash scripts/run_stage1.sh 
    ```

    <details>
    <summary> Explanation of "train_tag" and "val_tag" </summary>

    - Use `#` to seperate different datasets

    - Datasets:
      - `scanrefer`: [ScanRefer](https://github.com/daveredrum/ScanRefer) Dataset
      - `scan2cap`: [Scan2Cap](https://github.com/daveredrum/Scan2Cap) Dataset
      - `scanqa`: [ScanQA](https://github.com/ATR-DBI/ScanQA) Dataset
      - `sqa3d`: [SQA3D](https://github.com/SilongYong/SQA3D) Dataset
      - `multi3dref`: [Multi3dRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP) Dataset
      - `nr3d_caption`: A captioning dataset originated from [Nr3D](https://github.com/referit3d/referit3d).
      - `obj_align`: A dataset originated from ScanRefer to align the object identifiers with object tokens.
    
    - Please check the script file for further explanation of the other dataset.

    </details>
  
  - Stage2:
    ```
    bash scripts/run_stage2.sh 
    ```
  For each stage, we set the epoch as 3 but we manually stop the training after 2 epochs.

- Evaluate
  
  <!-- - Modify [run.sh](scripts/run.sh): () -->
  
    ```
    bash scripts/eval.sh
    ```
  
  We provide our checkpoint in [Google Drive](https://drive.google.com/drive/folders/14Si8bdWI3N5NEeVDLhmAlxilWPl0f_Wp?usp=sharing).
  

## 📄 Citation

If you find this project useful in your research, please consider cite:
```BibTeX
@article{kang2024robin3d,
  title={Robin3d: Improving 3d large language model via robust instruction tuning},
  author={Kang, Weitai and Huang, Haifeng and Shang, Yuzhang and Shah, Mubarak and Yan, Yan},
  journal={arXiv preprint arXiv:2410.00255},
  year={2024}
}
```

## 😊 Acknowledgement

Thanks to the open source of [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene/tree/dev)!
