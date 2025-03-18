import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import EmbeddingNet, Adapter, extract_style, synthesize_with_style
from dataset import SensorDataset
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def pretrain_embedding(epochs=10, batch_size=32, learning_rate=1e-3, device='cuda'):
    """
    Pre-trains the embedding network on a large-scale (source) dataset using classification loss.
    
    Returns:
        model: The pre-trained embedding network (E_theta).
    """
    # import a source dataset
    source_train_path = "/content/drive/MyDrive/FAS2AR/data/Source_train_1.csv"
    source_dataset = SensorDataset(source_train_path)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    
    num_classes = source_dataset.num_classes
    model = EmbeddingNet(input_channels=1).to(device)
    # A simple classification head for training
    classifier = nn.Linear(128, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)
    
    model.train()
    classifier.train()
    
    print("Starting pre-training on source data...")
    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in source_loader:
            x = x.to(device)  # (batch_size, channels, seq_length)
            y = y.to(device)
            optimizer.zero_grad()
            embedding = model(x)  # (batch_size, 128)
            logits = classifier(embedding)  # (batch_size, num_classes)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(source_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
    
    print("Pre-training complete.\n")
    return model, num_classes

def few_shot_adaptation(pretrained_model, num_classes,  epochs=10, batch_size=8, learning_rate=1e-3,
                          alpha=0.1, lambda_adv=0.5, device='cuda'):
    """
    Performs few-shot adaptation on target-domain data using adversarial style regularization.
    
    Args:
        pretrained_model: The frozen embedding network (E_theta).
        alpha: Step size for the adversarial style perturbation.
        lambda_adv: Weighting factor for the adversarial loss.
    
    Returns:
        adapter: The trained adapter network (phi).
        target_dataset: The target domain dataset (can be used for evaluation).
    """
    # Freeze the pre-trained embedding network
    for param in pretrained_model.parameters():
        param.requires_grad = False
    
    adapter = Adapter(embedding_dim=128, num_classes=num_classes).to(device)
    
    # import a few-shot target dataset (e.g., 27 samples)
    
    target_train_path = "/content/drive/MyDrive/FAS2AR/data/Target_train.csv"
    target_dataset = SensorDataset(target_train_path)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(adapter.parameters(), lr=learning_rate)
    
    pretrained_model.eval()  # Ensure the embedding is frozen
    
    print("Starting few-shot adaptation on target data...")
    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in target_loader:
            x = x.to(device)
            y = y.to(device)
            adapter.train()
            optimizer.zero_grad()
            
            # --------- Adversarial Style Generation ---------
            # Compute style vector v_s from x.
            # We want to compute gradients w.r.t. the style vector.
            v_s = extract_style(x)  # shape: (batch_size, channels*2)
            v_s.requires_grad = True  # enable gradient tracking on the style vector
            
            # Synthesize a baseline sample (x_hat) using the current style.
            # In an ideal case, g(x, v_s) recovers x.
            x_hat = synthesize_with_style(x, v_s)
            
            # Forward pass through the frozen embedding and adapter to compute loss.
            embedding_hat = pretrained_model(x_hat)
            logits_hat = adapter(embedding_hat)
            loss_hat = criterion(logits_hat, y)
            
            # Compute the gradient of loss_hat with respect to the style vector.
            grad_v = torch.autograd.grad(loss_hat, v_s, retain_graph=True)[0]
            # Compute the adversarial style vector: v_s_adv = v_s + alpha * sign(grad_v)
            v_s_adv = v_s + alpha * torch.sign(grad_v)
            # Synthesize the adversarial sample using v_s_adv.
            x_adv = synthesize_with_style(x, v_s_adv)
            # --------------------------------------------------
            
            # Compute loss on the original x (using frozen embedding)
            with torch.no_grad():
                embedding_orig = pretrained_model(x)
            logits_orig = adapter(embedding_orig)
            loss_orig = criterion(logits_orig, y)
            
            # Compute loss on the adversarial sample x_adv.
            embedding_adv = pretrained_model(x_adv)
            logits_adv = adapter(embedding_adv)
            loss_adv = criterion(logits_adv, y)
            
            # Total loss is a combination of the original and adversarial losses.
            loss_total = loss_orig + lambda_adv * loss_adv
            loss_total.backward()
            optimizer.step()
            
            running_loss += loss_total.item()
        
        avg_loss = running_loss / len(target_loader)
        print(f"Adaptation Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
    
    print("Few-shot adaptation complete.\n")
    return adapter

def evaluate(pretrained_model, adapter, dataset, batch_size=16, device='cuda', 
    class_names=None, plot_confusion=True):
    """
    Evaluates the adapted model on a given dataset, computing:
    - Average cross-entropy loss
    - Accuracy
    - Weighted F1 score
    Optionally plots a confusion matrix if class_names are provided.
    
    Args:
        pretrained_model: The frozen base embedding model (E_theta).
        adapter: The adapter network (phi) on top of the embedding.
        dataset: A PyTorch Dataset (or subclass) containing (features, labels).
        batch_size (int): Batch size for evaluation.
        device (str): 'cuda' or 'cpu'.
        class_names (list of str): Names of each class for confusion matrix.
        plot_confusion (bool): Whether to display a confusion matrix plot.
    
    Returns:
        avg_loss (float): Mean cross-entropy loss over the dataset.
        accuracy (float): Overall classification accuracy.
        f1_w (float): Weighted F1 score.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Collect predictions and labels for F1 and confusion matrix
    all_preds = []
    all_labels = []
    
    pretrained_model.eval()
    adapter.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            embedding = pretrained_model(x)
            logits = adapter(embedding)
            
            # Loss
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            
            # Predictions
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            
            # Store for F1/confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / total
    accuracy = correct / total
    f1_w = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Weighted F1: {f1_w:.4f}\n")
    
    # Optionally plot confusion matrix
    if plot_confusion and class_names is not None:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()
    
    return avg_loss, accuracy, f1_w