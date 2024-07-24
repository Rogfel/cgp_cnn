# https://gplearn.readthedocs.io/en/stable/index.html
import numpy as np
from gplearn.genetic import SymbolicTransformer


function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg',
                'inv', 'max', 'min']


def transformer():
    return SymbolicTransformer(generations=200, population_size=2000,
                               hall_of_fame=100, n_components=10,
                               function_set=function_set, parsimony_coefficient=0.0005,
                               max_samples=0.9, verbose=1, random_state=0, n_jobs=3)



if __name__ == '__main__':
    from sklearn.utils.validation import check_random_state
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import Ridge

    rng = check_random_state(0)
    diabetes = load_diabetes()
    perm = rng.permutation(diabetes.target.size)
    diabetes.data = diabetes.data[perm]
    diabetes.target = diabetes.target[perm]

    tgp = transformer()
    tgp.fit(diabetes.data[:300, :], diabetes.target[:300])
    
    gp_features = tgp.transform(diabetes.data)
    new_diabetes = np.hstack((diabetes.data, gp_features))

    est = Ridge()
    est.fit(new_diabetes[:300, :], diabetes.target[:300])
    print(est.score(new_diabetes[300:, :], diabetes.target[300:]))

