import json
import os

def patch_auc(filepath):
    if not os.path.exists(filepath):
        print(f"Skipping {filepath}, does not exist.")
        return
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        modified = False
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                new_source = source
                
                # Import
                if 'from sklearn.metrics import' in new_source and 'roc_auc_score' not in new_source:
                    new_source = new_source.replace(
                        'from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report',
                        'from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score'
                    )
                    modified = True

                # Patching StandaloneTrainer evaluating
                if 'class StandaloneTrainer:' in new_source and 'def evaluate(' in new_source:
                    if 'all_probs = []' not in new_source:
                        new_source = new_source.replace(
                            "all_preds, all_targets = [], []",
                            "all_preds, all_targets = [], []\n        all_probs = []"
                        )
                        new_source = new_source.replace(
                            "preds = torch.argmax(out[\"logits_ba\"], dim=1)",
                            "probs = torch.softmax(out[\"logits_ba\"], dim=1)\n            all_probs.extend(probs.cpu().numpy())\n            preds = torch.argmax(out[\"logits_ba\"], dim=1)"
                        )
                        auc_code = """
        all_probs_np = np.array(all_probs)
        try:
            if all_probs_np.shape[1] == 2:
                auc = roc_auc_score(all_targets, all_probs_np[:, 1])
            else:
                auc = roc_auc_score(all_targets, all_probs_np, multi_class='ovr', average='weighted')
        except Exception:
            auc = 0.0
            
        return total_loss / len(dataloader), f1, bAcc, all_preds, all_targets, auc"""
                        new_source = new_source.replace(
                            "return total_loss / len(dataloader), f1, bAcc, all_preds, all_targets",
                            auc_code[1:] # remove first newline
                        )
                        modified = True

                # Patching CausalTrainer evaluating (v2)
                if 'class CausalTrainer:' in new_source and 'def evaluate(' in new_source:
                    if 'all_probs = []' not in new_source:
                        new_source = new_source.replace(
                            "self.model.eval(); all_preds, all_labels = [], []; total_loss = 0",
                            "self.model.eval(); all_preds, all_labels = [], []; total_loss = 0\n        all_probs = []"
                        )
                        new_source = new_source.replace(
                            "preds = torch.argmax(logits, dim=1)",
                            "probs = torch.softmax(logits, dim=1)\n            all_probs.extend(probs.cpu().numpy())\n            preds = torch.argmax(logits, dim=1)"
                        )
                        
                        auc_code_v2 = """
        all_probs_np = np.array(all_probs)
        try:
            if all_probs_np.shape[1] == 2:
                auc = roc_auc_score(all_labels, all_probs_np[:, 1])
            else:
                auc = roc_auc_score(all_labels, all_probs_np, multi_class='ovr', average='weighted')
        except Exception:
            auc = 0.0
            
        if return_details:
            return total_loss / len(dataloader), f1, bacc, np.array(all_preds), np.array(all_labels), auc
        return total_loss / len(dataloader), f1, bacc, auc"""
                        # v2 returns 'return total_loss / len(dataloader), f1, bacc'
                        if "if return_details:\n            return total_loss / len(dataloader), f1, bacc, np.array(all_preds), np.array(all_labels)\n        return total_loss / len(dataloader), f1, bacc" in new_source:
                            new_source = new_source.replace(
                                "if return_details:\n            return total_loss / len(dataloader), f1, bacc, np.array(all_preds), np.array(all_labels)\n        return total_loss / len(dataloader), f1, bacc",
                                auc_code_v2[1:]
                            )
                        modified = True
                        
                cell['source'] = [new_source]
                
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print(f'Patched AUC in {filepath}')
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'Error patching {filepath}: {e}')

patch_auc('causal_crisis_standalone_colab.ipynb')
patch_auc('causal_crisis_v2_training.ipynb')
patch_auc('.worktrees/feature-cross-modal-attention/causal_crisis_v2_training.ipynb')
