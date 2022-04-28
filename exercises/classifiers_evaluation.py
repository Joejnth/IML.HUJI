from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from math import atan2, pi
from IMLearn.metrics import accuracy

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    df = np.load(filename)
    return df[:, [0, 1]], df[:, 2]


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        features, labels = load_dataset("C:/Users/t9134372/IML.HUJI/datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        perceptron = Perceptron()
        perceptron.fit(features, labels)
        losses = perceptron.callbacks[0]

        # Plot figure
        fig = go.Figure(
            layout=go.Layout(title="Perceptron Loss Per Iteration in the " + n + " Dataset", margin=dict(t=100)))
        fig.add_trace(go.Scatter(y=losses, mode='lines', name="Perceptron Loss"))
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        features, labels = load_dataset("C:/Users/t9134372/IML.HUJI/datasets/" + f)

        # Fit models and predict over training set

        lda = LDA()
        lda.fit(features, labels)
        lda_predictions = lda.predict(features)
        gnb = GaussianNaiveBayes()
        gnb.fit(features, labels)
        gnb_predictions = gnb.predict(features)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # from IMLearn.metrics import accuracy

        # accuracy, oval and x

        symbols = np.array(["circle", "x", "diamond"])
        fig = make_subplots(cols=2,
                            subplot_titles=["Gaussian Naive Bayes accuracy: " + str(accuracy(labels, gnb_predictions)),
                                            "LDA accuracy:" + str(accuracy(labels, lda_predictions))],
                            horizontal_spacing=0.01, vertical_spacing=.03)

        fig.update_layout(title="LDA and Gaussian Naive Bayes Plots", margin=dict(t=100))
        fig.add_trace(go.Scatter(x=features[:, 0], y=features[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=gnb_predictions, symbol=symbols[labels.astype(int)],
                                             line=dict(color="black", width=1))), row=1, col=1)
        fig.add_trace(go.Scatter(x=features[:, 0], y=features[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=lda_predictions, symbol=symbols[labels.astype(int)],
                                             line=dict(color="black", width=1))), row=1, col=2)
        # add X's and ovals
        fig.add_trace(go.Scatter(x=[lda.mu_[0][0], lda.mu_[1][0], lda.mu_[2][0]],
                                 y=[lda.mu_[0][1], lda.mu_[1][1], lda.mu_[2][1]], mode="markers",
                                 marker=dict(color="black", symbol=["x", "x", "x"], line=dict(color="black", width=1)),
                                 showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[gnb.mu_[0][0], gnb.mu_[1][0], gnb.mu_[2][0]],
                                 y=[gnb.mu_[0][1], gnb.mu_[1][1], gnb.mu_[2][1]], mode="markers",
                                 marker=dict(color="black", symbol=["x", "x", "x"], line=dict(color="black", width=1)),
                                 showlegend=False), row=1, col=1)

        fig.add_trace(go.Scatter(get_ellipse(gnb.mu_[0], np.diag(gnb.vars_[0])), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(get_ellipse(gnb.mu_[1], np.diag(gnb.vars_[1])), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(get_ellipse(gnb.mu_[2], np.diag(gnb.vars_[2])), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(get_ellipse(lda.mu_[0], lda.cov_), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(get_ellipse(lda.mu_[1], lda.cov_), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(get_ellipse(lda.mu_[2], lda.cov_), showlegend=False), row=1, col=2)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
