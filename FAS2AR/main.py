import torch
from train import pretrain_embedding, few_shot_adaptation, evaluate
from dataset import SensorDataset
import random
import numpy as np
def fix_seed(seed=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(2)

def main():
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    class_names = [
    "Eat", "Cook_Lunch", "Toilet", "Cook_Breakfast",
    "Cook_Dinner", "Bathe", "Sleep", "Relax", "Dress"
    ]
    
    # ------------------------------
    # 1. Pre-train Embedding Network
    # ------------------------------
    print("Pre-training the embedding network (source domain)...")
    embedding_model, num_classes = pretrain_embedding(epochs=5, batch_size=32, learning_rate=1e-3, device=device)
    
    # ------------------------------
    # 2. Few-Shot Adaptation with HAESAR
    # ------------------------------
    print("Performing few-shot adaptation (target domain)...")
    adapter = few_shot_adaptation(
        embedding_model,
        num_classes,
        epochs=10,
        batch_size=8,
        learning_rate=1e-3,
        alpha=0.1,
        lambda_adv=0.5,
        device=device
    )
    
    # ------------------------------
    # 3. Evaluation
    # ------------------------------
    target_test_path  = "/content/drive/MyDrive/FAS2AR/data/Target_test.csv"
    target_dataset = SensorDataset(target_test_path)
    print("Evaluating the adapted model on target data...")
    evaluate(embedding_model, adapter, target_dataset, batch_size=16, device=device, 
    class_names=class_names, plot_confusion=True)

if __name__ == '__main__':
    main()