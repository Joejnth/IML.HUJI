import math
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics.loss_functions import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_, counts = np.unique(y, return_counts=True)
        class_count = np.size(self.classes_)
        self.mu_ = np.zeros((class_count, np.shape(X)[1]))
        self.pi_ = np.zeros(class_count)
        for k in range(class_count):
            self.pi_[k] = counts[k] / np.size(y)
            self.mu_[k] = np.sum(X[y == self.classes_[k]], axis=0) / counts[k]  # unbiased estimator

        self.cov_ = np.zeros((np.shape(X)[1], np.shape(X)[1]))
        for i in range(np.size(y)):
            diff = X[i] - self.mu_[self.classes_ == y[i]]
            self.cov_ += np.matmul(np.transpose(diff), diff)
        self.cov_ /= np.shape(X)[0] - np.size(self.classes_)
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        probabilities = np.zeros(np.size(self.classes_))
        predictions = np.zeros(np.shape(X)[0])
        for i in range(np.shape(X)[0]):
            for k in range(np.size(self.classes_)):
                probabilities[k] = np.log(self.pi_[k]) + np.matmul(np.matmul(X[i], self._cov_inv),
                                                                   self.mu_[k]) - 0.5 * np.matmul(
                    np.matmul(self.mu_[k], self._cov_inv), self.mu_[k])
            predictions[i] = self.classes_[np.argmax(probabilities)]
        return predictions

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        log_likelihoods = np.zeros((np.shape(X)[0], np.size(self.classes_)))
        for i in range(np.shape(X)[0]):
            for k in range(np.size(self.classes_)):
                log_likelihoods[i, k] = np.log(self.pi_[k]) - 0.5 * np.shape(X)[1] * np.log(np.pi) - 0.5 * np.log(
                    det(self.cov_)) - 0.5 * np.matmul(np.matmul(X[i] - self.mu_[k], self._cov_inv), X[i] - self.mu_[k])
        return log_likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(self.predict(X), y)
