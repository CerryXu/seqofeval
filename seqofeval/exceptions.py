
class UndefinedMetricWarning(UserWarning):
    """Warning used when the metric is invalid

    .. versionchanged:: 0.18
       Moved from sklearn.base.
    """

class ConvergenceWarning(UserWarning):
    """Custom warning to capture convergence problems

    Examples
    --------

    >>> import numpy as np
    >>> import warnings
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.exceptions import ConvergenceWarning
    >>> warnings.simplefilter("always", ConvergenceWarning)
    >>> X = np.asarray([[0, 0],
    ...                 [0, 1],
    ...                 [1, 0],
    ...                 [1, 0]])  # last point is duplicated
    >>> with warnings.catch_warnings(record=True) as w:
    ...     km = KMeans(n_clusters=4).fit(X)
    ...     print(w[-1].message)
    Number of distinct clusters (3) found smaller than n_clusters (4).
    Possibly due to duplicate points in X.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.
    """
