"""
Baseline to compare with SRC: Eigenfaces + linear SVM, the classic method
A higher score means a more confident match, so it can be evaluated with the
same false accept and false reject curves as SRC
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC


class EigenfacesSVM:
    """PCA on the gallery, then a linear SVM. The score is the highest class probability"""

    def __init__(self, n_components=120, C=1.0, seed=0):
        self.n_components = n_components
        self.C = C
        self.seed = seed

    def fit(self, X, labels):
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.pca_ = PCA(n_components=n_comp, whiten=True, random_state=self.seed).fit(X)
        # The calibration turns SVM margins into probabilities, used as the score.
        self.svm_ = CalibratedClassifierCV(SVC(kernel="linear", C=self.C), ensemble=False, cv=3)
        self.svm_.fit(self.pca_.transform(X), labels)
        return self

    def predict_with_score(self, X):
        proba = self.svm_.predict_proba(self.pca_.transform(X))
        return self.svm_.classes_[proba.argmax(axis=1)], proba.max(axis=1)