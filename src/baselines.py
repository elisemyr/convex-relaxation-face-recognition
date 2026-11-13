"""
Baselines to compare with SRC

1. Eigenfaces + linear SVM: the classic method
2. Eigenfaces + nearest neighbour: the simplest possible method
3. Deep embeddings + nearest neighbour: a pretrained FaceNet model, which is
   the approach a company would most likely buy today

They all have the same interface as the SRC classifiers:
    fit(X, labels) and predict_with_score(X) -> pred, score
A higher score always means a more confident match, so every method can be
evaluated with the same false accept and false reject curves
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from .features import normalize_rows


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


class EigenfacesNN:
    """PCA on the gallery, then the nearest gallery face. The score is the cosine similarity"""

    def __init__(self, n_components=120, seed=0):
        self.n_components = n_components
        self.seed = seed

    def fit(self, X, labels):
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.pca_ = PCA(n_components=n_comp, random_state=self.seed).fit(X)
        self.G_ = normalize_rows(self.pca_.transform(X))
        self.labels_ = np.asarray(labels)
        return self

    def predict_with_score(self, X):
        sims = normalize_rows(self.pca_.transform(X)) @ self.G_.T
        best = sims.argmax(axis=1)
        return self.labels_[best], sims.max(axis=1)


class DeepEmbeddingNN:
    """
    Pretrained FaceNet, InceptionResnetV1 trained on VGGFace2, from facenet-pytorch

    Each face becomes a 512-dimensional vector. We match a test face to the
    gallery face with the highest cosine similarity, and use this similarity
    as the score.

    This class takes images of shape (k, h, w), not flat vectors, because the
    network needs the 2D image. Gray images are copied into 3 channels and
    resized to 160 x 160, the input size of the network

    The weights are downloaded the first time, about 100 MB
    """

    def __init__(self, device="cpu", batch_size=64):
        self.device = device
        self.batch_size = batch_size
        self.model_ = None

    def _load_model(self):
        if self.model_ is None:
            from facenet_pytorch import InceptionResnetV1

            self.model_ = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        return self.model_

    def embed(self, images):
        import torch
        import torch.nn.functional as F

        model = self._load_model()
        out = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = torch.as_tensor(np.asarray(images[start:start + self.batch_size]),
                                        dtype=torch.float32)
                batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)
                batch = F.interpolate(batch, size=(160, 160), mode="bilinear", align_corners=False)
                # Same scaling as facenet-pytorch: pixels in [0, 255], then (x - 127.5) / 128.
                batch = (batch * 255.0 - 127.5) / 128.0
                out.append(model(batch.to(self.device)).cpu().numpy())
        return normalize_rows(np.concatenate(out))

    def fit(self, images, labels):
        self.G_ = self.embed(images)
        self.labels_ = np.asarray(labels)
        return self

    def predict_with_score(self, images):
        sims = self.embed(images) @ self.G_.T
        best = sims.argmax(axis=1)
        return self.labels_[best], sims.max(axis=1)