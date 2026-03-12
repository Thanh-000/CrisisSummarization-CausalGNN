import json
import os

def patch_v2_calls(filepath):
    if not os.path.exists(filepath):
        return
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        modified = False
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                new_source = source

                if 'test_loss, test_f1, test_bacc = trainer.evaluate(test_loader)' in new_source and 'best_run_auc' not in new_source:
                    new_source = new_source.replace(
                        'best_run_f1 = 0\n    best_run_bacc = 0',
                        'best_run_f1 = 0\n    best_run_bacc = 0\n    best_run_auc = 0'
                    )
                    new_source = new_source.replace(
                        'test_loss, test_f1, test_bacc = trainer.evaluate(test_loader)',
                        'test_loss, test_f1, test_bacc, test_auc = trainer.evaluate(test_loader)'
                    )
                    new_source = new_source.replace(
                        'best_run_f1 = test_f1\n            best_run_bacc = test_bacc',
                        'best_run_f1 = test_f1\n            best_run_bacc = test_bacc\n            best_run_auc = test_auc'
                    )
                    new_source = new_source.replace(
                        "all_results_summary.append({'seed': s, 'f1': best_run_f1, 'bacc': best_run_bacc})",
                        "all_results_summary.append({'seed': s, 'f1': best_run_f1, 'bacc': best_run_bacc, 'auc': best_run_auc})"
                    )
                    new_source = new_source.replace(
                        'print(f"✅ Seed {s} Completed. Best F1: {best_run_f1:.4f} | Best bAcc: {best_run_bacc:.4f}")',
                        'print(f"✅ Seed {s} Completed. Best F1: {best_run_f1:.4f} | Best bAcc: {best_run_bacc:.4f} | Best AUC: {best_run_auc:.4f}")'
                    )
                    new_source = new_source.replace(
                        "bacc_list = [r['bacc'] for r in all_results_summary]",
                        "bacc_list = [r['bacc'] for r in all_results_summary]\nauc_list = [r['auc'] for r in all_results_summary]"
                    )
                    new_source = new_source.replace(
                        "mean_bacc, std_bacc = np.mean(bacc_list), np.std(bacc_list)",
                        "mean_bacc, std_bacc = np.mean(bacc_list), np.std(bacc_list)\nmean_auc, std_auc = np.mean(auc_list), np.std(auc_list)"
                    )
                    new_source = new_source.replace(
                        'print(f"Balanced Acc: {mean_bacc:.4f} ± {std_bacc:.4f}")',
                        'print(f"Balanced Acc: {mean_bacc:.4f} ± {std_bacc:.4f}")\nprint(f"AUC-ROC     : {mean_auc:.4f} ± {std_auc:.4f}")'
                    )
                    modified = True

                cell['source'] = [new_source]
                
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print(f'Patched V2 Calls in {filepath}')
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'Error patching {filepath}: {e}')

patch_v2_calls('causal_crisis_v2_training.ipynb')
patch_v2_calls('.worktrees/feature-cross-modal-attention/causal_crisis_v2_training.ipynb')
