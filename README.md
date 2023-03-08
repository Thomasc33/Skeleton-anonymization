# Replications

All pickle files generated from this repo are available on [google drive](https://drive.google.com/drive/folders/1PXky9r0hA1LfyTpXKwE4KHiCCmX9-1Df?usp=sharing)

### The conda environment used to compile Shift-GCN dependencies
```bash
conda env create -f shift_environment.yml
```

### Build Shift GCN Requirements
```bash
cd model/Temporal_shift
bash run.sh
```

### The main conda environment

```bash
conda env create -f environment.yml
```

### Generate data
Download the NTU60+120 skeletons from the link provided in [this repo](https://github.com/shahroudy/NTURGB-D)

[0-60](https://drive.google.com/open?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H), [60-120](https://drive.google.com/open?id=1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w)

Export that zip file to somewhere (Defualt is './data/nturgbd_raw')
```bash
python data_gen --data_path ./data/nturgbd_raw
```

### Executing the code
Run one of these. If you just want the pickling, then run the first two

Download [pretrained_unet.pt](https://drive.google.com/file/d/1U9NLJdkRVXBcXsT8lbyghJivhLci3L6M/view?usp=share_link) and put it in './save_models/'


```bash
python main.py --config ./config/test_adver_resnet.yaml --device 0
python main.py --config ./config/test_adver_unet.yaml --device 0
python main.py --config ./config/train_adver_unet.yaml --device 0
python main.py --config ./config/train_adver_resnet.yaml --device 0
```

# Modifications

- Made code actually run :) (There were some spacing errors/typos)
- Modified data code to operate on NTU120
- Pickle export the anonymizer models
- Pickle export test action and privacy loader
- Created "Run Anonymizer Model.ipynb" to get the anonymized skeleton in form of "X_unet.pkl" and "X_resnet.pkl"
