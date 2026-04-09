import numpy as np


def compute_sensitivity(
    model,
    X_test: np.ndarray,
    perturbation: float = 0.01,
    lag_column: int = 0,
) -> tuple[float, float]:
    """Finite-difference forecast sensitivity lambda_f.

    Parameters
    ----------
    model : object with .predict(X) method
    X_test : array, shape (n_samples, n_features)
    perturbation : float
        Relative perturbation size.
    lag_column : int
        Column index to perturb (default 0 = lag_1).

    Returns
    -------
    (mean_sensitivity, std_sensitivity)
    """
    base_pred = model.predict(X_test)
    X_perturbed = X_test.copy()
    delta = perturbation * np.abs(X_test[:, lag_column]).mean()
    if delta == 0:
        return 0.0, 0.0
    X_perturbed[:, lag_column] += delta
    perturbed_pred = model.predict(X_perturbed)
    sensitivities = np.abs(perturbed_pred - base_pred) / delta
    return float(sensitivities.mean()), float(sensitivities.std())
