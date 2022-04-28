from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_, counts = np.unique(y, return_counts=True)
        class_count = np.size(self.classes_)
        self.pi_ = np.zeros(class_count)
        self.mu_ = np.zeros((class_count, np.shape(X)[1]))
        self.vars_ = np.zeros((class_count, np.shape(X)[1]))

        for k in range(class_count):
            self.pi_[k] = counts[k] / np.size(y)
            self.mu_[k] = np.sum(X[y == self.classes_[k]], axis=0) / counts[k]
            for i in range(np.shape(X)[1]):
                self.vars_[k] = np.sum(np.square(X[y == self.classes_[k]] - self.mu_[k]), axis=0) / (
                        counts[k] - 1)  # unbiased estimator

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
        log_likelihood = self.likelihood(X)
        predictions = np.zeros(np.shape(X)[0])
        for i in range(np.shape(X)[0]):
            predictions[i] = self.classes_[np.argmax(log_likelihood[i])]
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
        for k in range(np.size(self.classes_)):
            class_likelihood = np.log(self.pi_[k]) - 0.5 * np.sum(np.log(self.vars_[k])) - 0.5 * np.log(2 * np.pi)
            for i in range(np.shape(X)[0]):
                diff = X[i] - self.mu_[k]
                log_likelihoods[i, k] = class_likelihood - 0.5 * np.sum(np.square(diff) / self.vars_[k])
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
