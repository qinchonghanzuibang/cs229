import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train, y_train)

    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    y_pred = model.predict(x_val)
    util.plot(x_val, y_val, model.theta, '{}.png'.format(pred_path))

    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        phi = 0
        for i in range(m):
            if y[i] == 1:
                phi += 1
        phi /= m
        # print(m, n, phi)
        mu_0_nu = 0
        mu_0_de = 0
        for i in range(m):
            if y[i] == 0:
                mu_0_de += 1
                mu_0_nu += x[i]
        mu_0 = mu_0_nu / mu_0_de
        # print(mu_0)
        mu_1_nu = 0
        mu_1_de = 0
        for i in range(m):
            if y[i] == 1:
                mu_1_de += 1
                mu_1_nu += x[i]
        mu_1 = mu_1_nu / mu_1_de
        # print(mu_1)
        sum_0 = np.zeros(n)
        sum_1 = np.zeros(n)
        count_0 = 0
        count_1 = 0
        for i in range(m):
            if y[i] == 0:
                sum_0 += x[i]
                count_0 += 1
            else:
                sum_1 += x[i]
                count_1 += 1
        mu_0 = sum_0 / count_0
        mu_1 = sum_1 / count_1

        cov_0 = np.zeros((n, n))
        cov_1 = np.zeros((n, n))

        for i in range(m):
            if y[i] == 0:
                diff = x[i] - mu_0
                cov_0 += np.outer(diff, diff)
            else:
                diff = x[i] - mu_1
                cov_1 += np.outer(diff, diff)

        Sigma = (cov_0 + cov_1) / m

        Sigma_inv = np.linalg.inv(Sigma)
        theta = Sigma_inv.dot(mu_1 - mu_0)
        theta_0 = 0.5 * (mu_0 + mu_1).T.dot(Sigma_inv).dot(mu_0 - mu_1) - np.log((1 - phi) / phi)

        self.theta = np.hstack([theta_0, theta])
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        x = util.add_intercept(x)
        p = 1 / (1 + np.exp(-x.dot(self.theta)))
        p[p >= 0.5] = 1
        p[p < 0.5] = 0
        return p
        # *** END CODE HERE
