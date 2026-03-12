import json
import os

def patch_calls(filepath):
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
                
                # Standalone unpacking
                if 'loss, test_f1, test_acc, _, _ = trainer.evaluate(' in new_source:
                    new_source = new_source.replace(
                        'test_loss, test_f1, test_acc, _, _ = trainer.evaluate(test_loader)',
                        'test_loss, test_f1, test_acc, _, _, test_auc = trainer.evaluate(test_loader)'
                    )
                    new_source = new_source.replace(
                        '| Balanced Acc: {test_acc:.4f}")',
                        '| bAcc: {test_acc:.4f} | AUC: {test_auc:.4f}")'
                    )
                    modified = True

                # v2 unpacking (multiple config modes C, N, G, I)
                if 'val_loss, val_f1, val_bacc = trainer.evaluate(' in new_source:
                    new_source = new_source.replace(
                        'val_loss, val_f1, val_bacc = trainer.evaluate(val_loader)',
                        'val_loss, val_f1, val_bacc, val_auc = trainer.evaluate(val_loader)'
                    )
                    new_source = new_source.replace(
                        '| bAcc: {val_bacc:.4f} | LR:',
                        '| bAcc: {val_bacc:.4f} | AUC: {val_auc:.4f} | LR:'
                    )
                    modified = True

                # Also returning test result
                if 'test_loss, test_f1, test_bacc = test_trainer.evaluate(' in new_source:
                    new_source = new_source.replace(
                        'test_loss, test_f1, test_bacc = test_trainer.evaluate(test_loader)',
                        'test_loss, test_f1, test_bacc, test_auc = test_trainer.evaluate(test_loader)'
                    )
                    new_source = new_source.replace(
                        '| bAcc: {test_bacc:.4f}\")',
                        '| bAcc: {test_bacc:.4f} | AUC: {test_auc:.4f}\")'
                    )
                    modified = True

                cell['source'] = [new_source]
                
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print(f'Patched Evaluate Calls in {filepath}')
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'Error patching {filepath}: {e}')

patch_calls('causal_crisis_standalone_colab.ipynb')
patch_calls('causal_crisis_v2_training.ipynb')
patch_calls('.worktrees/feature-cross-modal-attention/causal_crisis_v2_training.ipynb')
