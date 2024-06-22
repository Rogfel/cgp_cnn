from sklearn.utils.validation import check_random_state
from sklearn.datasets import load_breast_cancer
from gplearn.genetic import SymbolicClassifier
from sklearn.metrics import roc_auc_score
import graphviz


rng = check_random_state(0)
cancer = load_breast_cancer()
perm = rng.permutation(cancer.target.size)
cancer.data = cancer.data[perm]
cancer.target = cancer.target[perm]

est = SymbolicClassifier(parsimony_coefficient=.01,
                         feature_names=cancer.feature_names,
                         random_state=1)
print(est.fit(cancer.data[:400], cancer.target[:400]))

y_true = cancer.target[400:]
y_score = est.predict_proba(cancer.data[400:])[:,1]
print(roc_auc_score(y_true, y_score))

dot_data = est._program.export_graphviz()
graph = graphviz.Source(dot_data)
graph