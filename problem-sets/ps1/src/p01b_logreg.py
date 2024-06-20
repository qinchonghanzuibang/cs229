import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    model = LogisticRegression()
    model.fit(x_train, y_train)

    util.plot(x_train, y_train, model.theta, './output/p01b_plot.png')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        m, n = x.shape  # m is number of training examples, n is number of features
        self.theta = np.zeros(n)  # initialize theta to zero vector

        while True:
            theta_prev = np.copy(self.theta)
            h_theta_x = 1 / (1 + np.exp(-x.dot(self.theta)))
            H = (np.transpose(x) * h_theta_x * (1 - h_theta_x)).dot(x) / m

            gradient = np.transpose(x).dot(h_theta_x - y) / m
            H_inverse = np.linalg.inv(H)

            self.theta -= H_inverse.dot(gradient)

            if np.linalg.norm(self.theta - theta_prev, ord=1) < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        result = 1 / (1 + np.exp(-x.dot(self.theta)))
        return result

        # *** END CODE HERE ***
