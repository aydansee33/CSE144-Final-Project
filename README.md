# CSE 144 Final Project — Transfer Learning Challenge

**Team Members:** Aydan See
**Kaggle Score:** 0.772  
**Model Weights:** [Google Drive Link](https://drive.google.com/drive/folders/1LfPn9oDLHQtV5dm-5OisGqbG-eQ5ZsjU?usp=drive_link)

## Leaderboard
<img width="1139" height="640" alt="Screenshot 2026-06-10 at 9 47 17 PM" src="https://github.com/user-attachments/assets/c0e01026-dc33-4a57-8566-7921cc3e96f1" />


## Setup
\```bash
pip install torch torchvision
\```

## Training
1. Open `CSE144_Final.ipynb` in Google Colab
2. Mount Google Drive and set `TRAIN_DIR` and `TEST_DIR` to your data paths
3. Run all cells in order

The training pipeline:
- EfficientNet-B0, B3, B4 fine-tuned with AdamW, lr=1e-4, weight_decay=1e-2
- ResNet50, ResNet101 fine-tuned with same settings
- Vision Transformer (ViT-B/16) fine-tuned with same settings
- All models trained for 50 epochs with CosineAnnealingLR
- Label smoothing 0.1, batch size 32, image size 224x224
- Random seed: 42

## Inference
1. Load all 6 model weights from Google Drive into Colab
2. Run the ensemble submission cell
3. Output saved to `submission_6model.csv`

## Reproducibility
- Random seed: 42 set for torch, numpy, and random
- `torch.backends.cudnn.deterministic = True`
- All weights saved to Google Drive (link above)
