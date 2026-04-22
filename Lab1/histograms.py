import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform

distributions = {
    'Нормальное': {
        'rvs': lambda size: np.random.normal(0, 1, size),
        'pdf': lambda x: norm.pdf(x, 0, 1),
        'x_range': (-4, 4),
        'is_discrete': False
    },
    'Коши': {
        'rvs': lambda size: np.random.standard_cauchy(size),
        'pdf': lambda x: cauchy.pdf(x, 0, 1),
        'x_range': (-10, 10),
        'is_discrete': False
    },
    'Лапласа': {
        'rvs': lambda size: np.random.laplace(0, 1 / np.sqrt(2), size),
        'pdf': lambda x: laplace.pdf(x, 0, 1 / np.sqrt(2)),
        'x_range': (-6, 6),
        'is_discrete': False
    },
    'Пуассона': {
        'rvs': lambda size: np.random.poisson(10, size),
        'pmf': lambda k: poisson.pmf(k, 10),
        'x_range': (0, 20),
        'is_discrete': True
    },
    'Равномерное': {
        'rvs': lambda size: np.random.uniform(-np.sqrt(3), np.sqrt(3), size),
        'pdf': lambda x: uniform.pdf(x, -np.sqrt(3), 2 * np.sqrt(3)),
        'x_range': (-2.5, 2.5),
        'is_discrete': False
    }
}

sample_sizes = [10, 100, 1000]

for dist_name, params in distributions.items():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Распределение: {dist_name}', fontsize=16)

    for idx, n in enumerate(sample_sizes):
        sample = params['rvs'](n)

        ax = axes[idx]
        if params['is_discrete']:
            bins = np.arange(-0.5, 21.5, 1)
            ax.hist(sample, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Выборка')
            k_vals = np.arange(0, 21)
            prob_theor = params['pmf'](k_vals)
            ax.plot(k_vals, prob_theor, 'ro-', markersize=4, label='Теоретическая PMF')
            ax.set_xlim(-0.5, 20.5)
        else:
            ax.hist(sample, bins='auto', density=True, alpha=0.6, color='skyblue', edgecolor='black',
                    label='Гистограмма')
            x_vals = np.linspace(params['x_range'][0], params['x_range'][1], 500)
            pdf_theor = params['pdf'](x_vals)
            ax.plot(x_vals, pdf_theor, 'r-', linewidth=2, label='Теоретическая плотность')
            ax.set_xlim(params['x_range'])

        ax.set_title(f'n = {n}')
        ax.set_xlabel('x')
        ax.set_ylabel('Плотность / Вероятность')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
