"""Pre-rank feature vector: (BM25 score, dense score, doc length), z-score
normalized before the MLP sees them -- the three raw values are on
incompatible scales (BM25 unbounded, dense cosine in [-1, 1], doc length in
the hundreds to thousands of characters), and an unnormalized MLP would
either fail to learn or be dominated by whichever feature happens to have
the largest raw magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

FeatureVector = tuple[float, float, float]


def build_feature_vector(bm25_score: float, dense_score: float, doc_length: int) -> FeatureVector:
    return (float(bm25_score), float(dense_score), float(doc_length))


@dataclass
class FeatureScaler:
    mean: FeatureVector
    std: FeatureVector

    def transform(self, vector: FeatureVector) -> FeatureVector:
        return tuple(
            0.0 if std == 0.0 else (value - mean) / std
            for value, mean, std in zip(vector, self.mean, self.std)
        )


def fit_scaler(feature_vectors: list[FeatureVector]) -> FeatureScaler:
    n = len(feature_vectors)
    means = tuple(sum(v[i] for v in feature_vectors) / n for i in range(3))
    variances = tuple(
        sum((v[i] - means[i]) ** 2 for v in feature_vectors) / n for i in range(3)
    )
    stds = tuple(variance**0.5 for variance in variances)
    return FeatureScaler(mean=means, std=stds)
