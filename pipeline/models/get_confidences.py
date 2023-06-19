import numpy as np
import pickle 
from matplotlib import pyplot as plt 
import json 
import warnings 

def plot_confidence(ci, pi, ax, drug):
    """
    Make a bar plot indicating mean prediction pi and confidence intervals
    ci
    """
    for i, (c, m) in enumerate(zip(ci, pi)): # confidence, mean
        ax.plot([i, i], c, color='blue', linewidth=2)
        ax.plot([i-.1, i+.1], [c[0]]*2, color='blue', linewidth=2)
        ax.plot([i-.1, i+.1], [c[1]]*2, color='blue', linewidth=2)
        ax.plot([i], [m], '.', color='red', markersize=10)
    ax.set_ylim([-.1, 1.1])
    ax.set_xlabel('Class')
    ax.set_title(f'Estimated Fraction Drug {drug}')
    ax.grid()
    return ax

def get_confidence_computation(preds, C, cm, sample_size_mu, drug, alpha=.1, method=None):
    """
    Compute confidence intervals for the effectiveness of a drug
    on each class given the confusion matrix mu, using a specified
    method. If the method is None, we choose the general strategy
    (Lemma 1) if there are 2 predicted classes, and the improved
    strategy of Lemma 2 otherwise. We can also set 
    method = 'deterministic' to use Corollary 1.

    Parameters:
        preds:
            predicions list
        C:
            number of classes
        cm:
            confusion matrix
        sample_size_mu:
            number of samples used to estimate the confusion matrix
        drug:
            index of the drug for which we compute
        method:
            one of None, 'deterministic', or 'general'
        alpha:
            we compute a 1-alpha confidence interval 

    Returns:
        dictionary with keys 'ci' and 'mean', where 'ci' contains
        confidence intervals for the fraction of each label 
        (e.g. [.5, .7] label 1 - dead and [.3 - .5] label 0 - alive),
        and 'mean' is the "most likely" result, which always lies
        in the confidence interval
    """
    
    if method is None:
        method = 'lemma2' if C == 2 else 'general'
    
    one_hot = np.stack([np.arange(C) == p for p in preds], axis=0)
    counts = np.sum(one_hot, axis=0)
    # check counts for 0's
    nonzero = np.argwhere(counts != 0)
    zero_idx = np.argwhere(counts == 0)
    if C > 2:
        s = np.min(counts[nonzero]) # least frequently predicted label
    else:
        s = np.max(counts[nonzero]) # least frequently predicted label
    

    if method != 'deterministic': s = min([sample_size_mu, s])
    
    ts = {
        'general': np.sqrt((C+1)**2 / (2*s) * np.log(4*C / alpha)), 
        'lemma2': np.sqrt((2 / s) * np.log(6 / alpha)), 
        'deterministic': np.sqrt(1 / (2*s) * np.log(6 / alpha)), 
    }

    t = ts[method]

    nu = np.mean(one_hot, axis=0)
    pi = cm @ nu 
    
    ci = np.stack(
        [
            np.where(pi-t > 0, pi-t, 0), 
            np.where(pi+t < 1, pi+t, 1)
            # max([0, pi - t]), 
            # min([1, pi + t])
        ], 
        axis=1
    )

    # replace 0 idxs:
    for idx in zero_idx:
        print(f'Warning! Classifier predicted class {idx} zero times for drug {drug}, and this class will be disregarded in the computation')
        ci[idx] = np.array([0, 1])
        pi[idx] = np.nan 


    return {'ci': ci, 'mean': pi}

def get_confidence(pred_dir, confusion_dir, alpha=.1, method=None, plot=True):
    # group by
    with open(pred_dir) as f:
        pred_dict = json.load(f)
    with open(confusion_dir, 'rb') as f:
        confmat = pickle.load(f)
    cm = confmat['matrix']
    sample_size_mu = confmat['sample_size']
    cm = cm / cm.sum(axis=0)

    pd = [(p['prediction'], p['drug']) for p in pred_dict['predictions']]# preds, drugs
    n_drugs = max([i[-1] for i in pd]) + 1

    preds_per_drug = {d: [i[0] for i in pd if i[-1] == d] for d in range(n_drugs)}
    C = cm.shape[0]
    outputs = {}

    for d in range(n_drugs):
        outputs[d] = get_confidence_computation(preds_per_drug[d], C, cm, sample_size_mu, alpha=alpha, method=method, drug=d)
    
    if plot:
        fig, axs = plt.subplots(n_drugs, 1)
        if not isinstance(axs, np.ndarray): axs = [axs]
        for (drug, v), ax in zip(outputs.items(), axs):
            ci = v['ci']
            pi = v['mean']
            ax = plot_confidence(ci, pi, ax, drug)
        
        fig.tight_layout()
        return (outputs, (fig, axs))
    return outputs 