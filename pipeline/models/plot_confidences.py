import numpy as np 
from matplotlib import pyplot as plt 
import argparse 

def main():
    """
    Plot neccessary sample sizes for user-specified input sizes
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--C', help='number of classes for which to compute confidences', default=2)
    args = parser.parse_args()

    methods = ['general', 'lemma2', 'corollary1']
    C = int(args.C)

    ss = {
        'general': lambda a, t: (C+1)**2 * np.log(4*C / a) / (2*t**2),
        'lemma2': lambda a, t: 2*np.log(6 / a) / (t**2),
        'corollary1': lambda a, t: np.log(6 / a) / (2*t**2) # sample size
    }
    titles = {
        'general': 'General bounds', 
        'lemma2': 'Bounds as in Lemma 2', 
        'corollary1': 'Bounds as in Corollary 1'
    }
    
    h = 4
    fig, axs = plt.subplots(1,3, figsize=(3*h, h))
    
    if C != 2: axs = [axs[0]] # the other methods are only appropriate if C=2

    x = np.linspace(0.02, 0.5, 100)
    for ax, method in zip(axs, methods):
        s = ss[method]
        for a in np.linspace(.1, 1, 10):
            ax.plot(x, [s(a, xi) for xi in x], label=f'{round(a, 2)}')
        
        ax.legend(title=r'$\alpha$')
        plt.rcParams['text.usetex'] = True
        ax.set_xlabel('$c_{d, k}(\\alpha)$')
        plt.rcParams['text.usetex'] = False
        ax.set_ylabel('Sample Size s')
        ax.set_ylim([0, 1200])
        ax.set_title(titles[method])
        ax.grid()
    fig.suptitle(f'Deviation vs. sample size for different confidences for C = {C}')
    fig.tight_layout()
    fig.savefig(f'confidences_{C=}.png')

if __name__ == '__main__':
    main()