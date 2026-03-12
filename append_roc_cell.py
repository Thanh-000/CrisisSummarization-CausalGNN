import json
import os

def append_roc_cell(filepath):
    if not os.path.exists(filepath):
        return
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        # Check if already added
        for cell in nb['cells']:
            if cell['cell_type'] == 'code' and 'ROC CURVE ANALYSIS' in ''.join(cell['source']):
                print(f"ROC cell already exists in {filepath}")
                return

        roc_code = """## 📈 10. ROC CURVE ANALYSIS
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_roc_curve(model, dataloader, num_classes, trainer_instance=None):
    model.eval()
    all_probs = []
    all_labels = []

    print("🔍 Collecting probabilities for ROC Curve...")
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                img, txt, labels, _ = [b.to(device) for b in batch]
            else:
                img, txt, labels = [b.to(device) for b in batch]
                
            out_pre = model(img, txt)
            adj = build_knn_graph(out_pre['xc'], k=3)
            
            # Get backdoor_xs
            sampled_xs = None
            if trainer_instance is hasattr(trainer_instance, 'mem_bank'):
                sampled_xs = trainer_instance.mem_bank.sample(50)
            elif hasattr(trainer_instance, 'memory_bank'):
                bank_samples = trainer_instance.memory_bank.sample(M=4)
                sampled_xs = bank_samples.unsqueeze(0).expand(img.size(0), 4, -1)
                
            out = model(img, txt, adj=adj, backdoor_xs=sampled_xs)
            probs = torch.softmax(out.get('logits_ba', out['logits']), dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    plt.figure(figsize=(10, 8))
    
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    else:
        # Binarize the output
        labels_bin = label_binarize(all_labels, classes=range(num_classes))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve of class {i} (area = {roc_auc:.4f})')
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    
    task_name = CURRENT_TASK.upper() if 'CURRENT_TASK' in globals() else "TASK"
    plt.title(f'Receiver Operating Characteristic (ROC) - {task_name}', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.2)
    plt.show()

# Run ROC evaluation using the latest trained model & trainer object
if 'trainer' in globals():
    plot_roc_curve(model, test_loader, num_classes=NUM_CLASSES if 'NUM_CLASSES' in globals() else num_classes, trainer_instance=trainer)
"""
        
        # Split by newlines but keep the \n at the end of each line
        source_lines = [line + '\n' for line in roc_code.split('\n')]
        # remove last extra newline if present
        source_lines[-1] = source_lines[-1].strip('\n')
        
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines
        }
        
        nb['cells'].append(new_cell)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f'Appended ROC Evaluation cell to {filepath}')
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'Error patching {filepath}: {e}')

append_roc_cell('causal_crisis_v2_training.ipynb')
append_roc_cell('.worktrees/feature-cross-modal-attention/causal_crisis_v2_training.ipynb')
