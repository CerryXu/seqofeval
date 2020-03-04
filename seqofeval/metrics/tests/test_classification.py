import warnings

import numpy as np
import pytest
from seqeval.metrics import precision_score, recall_score

from seqofeval import datasets
from seqofeval.exceptions import UndefinedMetricWarning
from seqofeval.metrics._classification import accuracy_score, fbeta_score, balanced_accuracy_score
from seqofeval.metrics._classification import recall_score
from seqofeval.metrics._classification import classification_report

from seqofeval.metrics._classification import f1_score

from seqofeval.metrics._classification import precision_recall_fscore_support
from seqofeval.metrics._ranking import average_precision_score
from seqofeval.utils._testing import assert_array_equal, ignore_warnings, assert_array_almost_equal,assert_almost_equal, assert_no_warnings, assert_warns_message


# Utilities for testing


# def make_prediction(dataset=None, binary=False):
#     """Make some classification predictions on a toy dataset using a SVC
#
#     If binary is True restrict to a binary classification problem instead of a
#     multiclass classification problem
#     """
#
#     if dataset is None:
#         # import some data to play with
#         dataset = datasets.load_iris()
#
#     X = dataset.data
#     y = dataset.target
#
#     if binary:
#         # restrict to a binary classification task
#         X, y = X[y < 2], y[y < 2]
#
#     n_samples, n_features = X.shape
#     p = np.arange(n_samples)
#
#     rng = check_random_state(37)
#     rng.shuffle(p)
#     X, y = X[p], y[p]
#     half = int(n_samples / 2)
#
#     # add noisy features to make the problem harder and avoid perfect results
#     rng = np.random.RandomState(0)
#     X = np.c_[X, rng.randn(n_samples, 200 * n_features)]
#
#     # run classifier, get class probabilities and label predictions
#     clf = svm.SVC(kernel='linear', probability=True, random_state=0)
#     probas_pred = clf.fit(X[:half], y[:half]).predict_proba(X[half:])
#
#     if binary:
#         # only interested in probabilities of the positive case
#         # XXX: do we really want a special API for the binary case?
#         probas_pred = probas_pred[:, 1]
#
#     y_pred = clf.predict(X[half:])
#     y_true = y[half:]
#     return y_true, y_pred, probas_pred
#
#

# Test  accuracy_score begin
from seqofeval.utils.validation import check_random_state


def test_multilabel_accuracy_score_subset_accuracy():
    # Dense label indicator matrix format
    y1 = np.array([[0, 1, 1], [1, 0, 1]])
    y2 = np.array([[0, 0, 1], [1, 0, 1]])

    assert accuracy_score(y1, y2) == 0.5
    assert accuracy_score(y1, y1) == 1
    assert accuracy_score(y2, y2) == 1
    assert accuracy_score(y2, np.logical_not(y2)) == 0
    assert accuracy_score(y1, np.logical_not(y1)) == 0
    assert accuracy_score(y1, np.zeros(y1.shape)) == 0
    assert accuracy_score(y2, np.zeros(y1.shape)) == 0



def test_balanced_accuracy_score_unseen():
    assert_warns_message(UserWarning, 'y_pred contains classes not in y_true',
                         balanced_accuracy_score, [0, 0, 0], [0, 0, 1])


@pytest.mark.parametrize('y_true,y_pred',
                         [
                             (['a', 'b', 'a', 'b'], ['a', 'a', 'a', 'b']),
                             (['a', 'b', 'c', 'b'], ['a', 'a', 'a', 'b']),
                             (['a', 'a', 'a', 'b'], ['a', 'b', 'c', 'b']),
                         ])
def test_balanced_accuracy_score(y_true, y_pred):
    macro_recall = recall_score(y_true, y_pred, average='macro',
                                labels=np.unique(y_true))
    with ignore_warnings():
        # Warnings are tested in test_balanced_accuracy_score_unseen
        balanced = balanced_accuracy_score(y_true, y_pred)
    assert balanced == pytest.approx(macro_recall)
    assert balanced == pytest.approx(macro_recall)
    adjusted = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    chance = balanced_accuracy_score(y_true, np.full_like(y_true, y_true[0]))
    assert adjusted == (balanced - chance) / (1 - chance)

# Test accuracy_score end





# Test f1_score begin

# def test_precision_recall_f1_score_binary():
#     # Test Precision Recall and F1 Score for binary classification task
#     y_true, y_pred, _ = make_prediction(binary=True)
#
#     # detailed measures for each class
#     p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
#     assert_array_almost_equal(p, [0.73, 0.85], 2)
#     assert_array_almost_equal(r, [0.88, 0.68], 2)
#     assert_array_almost_equal(f, [0.80, 0.76], 2)
#     assert_array_equal(s, [25, 25])
#
#     # individual scoring function that can be used for grid search: in the
#     # binary class case the score is the value of the measure for the positive
#     # class (e.g. label == 1). This is deprecated for average != 'binary'.
#     for kwargs, my_assert in [({}, assert_no_warnings),
#                               ({'average': 'binary'}, assert_no_warnings)]:
#         ps = my_assert(precision_score, y_true, y_pred, **kwargs)
#         assert_array_almost_equal(ps, 0.85, 2)
#
#         rs = my_assert(recall_score, y_true, y_pred, **kwargs)
#         assert_array_almost_equal(rs, 0.68, 2)
#
#         fs = my_assert(f1_score, y_true, y_pred, **kwargs)
#         assert_array_almost_equal(fs, 0.76, 2)
#
#         assert_almost_equal(my_assert(fbeta_score, y_true, y_pred, beta=2,
#                                       **kwargs),
#                             (1 + 2 ** 2) * ps * rs / (2 ** 2 * ps + rs), 2)


# def test_precision_recall_f1_score_multiclass():
#     # Test Precision Recall and F1 Score for multiclass classification task
#     y_true, y_pred, _ = make_prediction(binary=False)
#
#     # compute scores with default labels introspection
#     p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
#     assert_array_almost_equal(p, [0.83, 0.33, 0.42], 2)
#     assert_array_almost_equal(r, [0.79, 0.09, 0.90], 2)
#     assert_array_almost_equal(f, [0.81, 0.15, 0.57], 2)
#     assert_array_equal(s, [24, 31, 20])
#
#     # averaging tests
#     ps = precision_score(y_true, y_pred, pos_label=1, average='micro')
#     assert_array_almost_equal(ps, 0.53, 2)
#
#     rs = recall_score(y_true, y_pred, average='micro')
#     assert_array_almost_equal(rs, 0.53, 2)
#
#     fs = f1_score(y_true, y_pred, average='micro')
#     assert_array_almost_equal(fs, 0.53, 2)
#
#     ps = precision_score(y_true, y_pred, average='macro')
#     assert_array_almost_equal(ps, 0.53, 2)
#
#     rs = recall_score(y_true, y_pred, average='macro')
#     assert_array_almost_equal(rs, 0.60, 2)
#
#     fs = f1_score(y_true, y_pred, average='macro')
#     assert_array_almost_equal(fs, 0.51, 2)
#
#     ps = precision_score(y_true, y_pred, average='weighted')
#     assert_array_almost_equal(ps, 0.51, 2)
#
#     rs = recall_score(y_true, y_pred, average='weighted')
#     assert_array_almost_equal(rs, 0.53, 2)
#
#     fs = f1_score(y_true, y_pred, average='weighted')
#     assert_array_almost_equal(fs, 0.47, 2)
#
#     with pytest.raises(ValueError):
#         precision_score(y_true, y_pred, average="samples")
#     with pytest.raises(ValueError):
#         recall_score(y_true, y_pred, average="samples")
#     with pytest.raises(ValueError):
#         f1_score(y_true, y_pred, average="samples")
#     with pytest.raises(ValueError):
#         fbeta_score(y_true, y_pred, average="samples", beta=0.5)
#
#     # same prediction but with and explicit label ordering
#     p, r, f, s = precision_recall_fscore_support(
#         y_true, y_pred, labels=[0, 2, 1], average=None)
#     assert_array_almost_equal(p, [0.83, 0.41, 0.33], 2)
#     assert_array_almost_equal(r, [0.79, 0.90, 0.10], 2)
#     assert_array_almost_equal(f, [0.81, 0.57, 0.15], 2)
#     assert_array_equal(s, [24, 20, 31])
#
@pytest.mark.parametrize('beta', [1])
@pytest.mark.parametrize('average', ["macro", "micro", "weighted", "samples"])
@pytest.mark.parametrize('zero_division', [0, 1])
def test_precision_recall_f1_no_labels(beta, average, zero_division):
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    p, r, f, s = assert_no_warnings(precision_recall_fscore_support, y_true,
                                    y_pred, average=average, beta=beta,
                                    zero_division=zero_division)
    fbeta = assert_no_warnings(fbeta_score, y_true, y_pred, beta=beta,
                               average=average, zero_division=zero_division)

    zero_division = float(zero_division)
    assert_almost_equal(p, zero_division)
    assert_almost_equal(r, zero_division)
    assert_almost_equal(f, zero_division)
    assert s is None

    assert_almost_equal(fbeta, float(zero_division))

@pytest.mark.parametrize('average', ["macro", "micro", "weighted", "samples"])
def test_precision_recall_f1_no_labels_check_warnings(average):
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    func = precision_recall_fscore_support
    with pytest.warns(UndefinedMetricWarning):
        p, r, f, s = func(y_true, y_pred, average=average, beta=1.0)

    assert_almost_equal(p, 0)
    assert_almost_equal(r, 0)
    assert_almost_equal(f, 0)
    assert s is None

    with pytest.warns(UndefinedMetricWarning):
        fbeta = fbeta_score(y_true, y_pred, average=average, beta=1.0)

    assert_almost_equal(fbeta, 0)

@pytest.mark.parametrize('zero_division', [0, 1])
def test_precision_recall_f1_no_labels_average_none(zero_division):
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    # tp = [0, 0, 0]
    # fn = [0, 0, 0]
    # fp = [0, 0, 0]
    # support = [0, 0, 0]
    # |y_hat_i inter y_i | = [0, 0, 0]
    # |y_i| = [0, 0, 0]
    # |y_hat_i| = [0, 0, 0]

    p, r, f, s = assert_no_warnings(precision_recall_fscore_support,
                                    y_true, y_pred,
                                    average=None, beta=1.0,
                                    zero_division=zero_division)
    fbeta = assert_no_warnings(fbeta_score, y_true, y_pred, beta=1.0,
                               average=None, zero_division=zero_division)

    zero_division = float(zero_division)
    assert_array_almost_equal(
        p, [zero_division, zero_division, zero_division], 2
    )
    assert_array_almost_equal(
        r, [zero_division, zero_division, zero_division], 2
    )
    assert_array_almost_equal(
        f, [zero_division, zero_division, zero_division], 2
    )
    assert_array_almost_equal(s, [0, 0, 0], 2)

    assert_array_almost_equal(
        fbeta, [zero_division, zero_division, zero_division], 2
    )


def test_precision_recall_f1_no_labels_average_none_warn():
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    # tp = [0, 0, 0]
    # fn = [0, 0, 0]
    # fp = [0, 0, 0]
    # support = [0, 0, 0]
    # |y_hat_i inter y_i | = [0, 0, 0]
    # |y_i| = [0, 0, 0]
    # |y_hat_i| = [0, 0, 0]

    with pytest.warns(UndefinedMetricWarning):
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, average=None, beta=1
        )

    assert_array_almost_equal(p, [0, 0, 0], 2)
    assert_array_almost_equal(r, [0, 0, 0], 2)
    assert_array_almost_equal(f, [0, 0, 0], 2)
    assert_array_almost_equal(s, [0, 0, 0], 2)

    with pytest.warns(UndefinedMetricWarning):
        fbeta = fbeta_score(y_true, y_pred, beta=1, average=None)

    assert_array_almost_equal(fbeta, [0, 0, 0], 2)

@pytest.mark.parametrize('average',
                         ['samples', 'micro', 'macro', 'weighted', None])
def test_precision_refcall_f1_score_multilabel_unordered_labels(average):
    # test that labels need not be sorted in the multilabel case
    y_true = np.array([[1, 1, 0, 0]])
    y_pred = np.array([[0, 0, 1, 1]])
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[3, 0, 1, 2], warn_for=[], average=average)
    assert_array_equal(p, 0)
    assert_array_equal(r, 0)
    assert_array_equal(f, 0)
    if average is None:
        assert_array_equal(s, [0, 1, 1, 0])


def test_precision_recall_f1_score_binary_averaged():
    y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1])

    # compute scores with default labels introspection
    ps, rs, fs, _ = precision_recall_fscore_support(y_true, y_pred,
                                                    average=None)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 average='macro')
    assert p == np.mean(ps)
    assert r == np.mean(rs)
    assert f == np.mean(fs)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 average='weighted')
    support = np.bincount(y_true)
    assert p == np.average(ps, weights=support)
    assert r == np.average(rs, weights=support)
    assert f == np.average(fs, weights=support)

@ignore_warnings
def test_precision_recall_f1_score_multilabel_1():
    # Test precision_recall_f1_score on a crafted multilabel example
    # First crafted example

    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    y_pred = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0]])

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)

    # tp = [0, 1, 1, 0]
    # fn = [1, 0, 0, 1]
    # fp = [1, 1, 0, 0]
    # Check per class

    assert_array_almost_equal(p, [0.0, 0.5, 1.0, 0.0], 2)
    assert_array_almost_equal(r, [0.0, 1.0, 1.0, 0.0], 2)
    assert_array_almost_equal(f, [0.0, 1 / 1.5, 1, 0.0], 2)
    assert_array_almost_equal(s, [1, 1, 1, 1], 2)

    f2 = fbeta_score(y_true, y_pred, beta=2, average=None)
    support = s
    assert_array_almost_equal(f2, [0, 0.83, 1, 0], 2)

    # Check macro
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average="macro")
    assert_almost_equal(p, 1.5 / 4)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 2.5 / 1.5 * 0.25)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2, average="macro"),
                        np.mean(f2))

    # Check micro
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average="micro")
    assert_almost_equal(p, 0.5)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 0.5)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2,
                                    average="micro"),
                        (1 + 4) * p * r / (4 * p + r))

    # Check weighted
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average="weighted")
    assert_almost_equal(p, 1.5 / 4)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 2.5 / 1.5 * 0.25)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2,
                                    average="weighted"),
                        np.average(f2, weights=support))
    # Check samples
    # |h(x_i) inter y_i | = [0, 1, 1]
    # |y_i| = [1, 1, 2]
    # |h(x_i)| = [1, 1, 2]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average="samples")
    assert_almost_equal(p, 0.5)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 0.5)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2, average="samples"),
                        0.5)

@ignore_warnings
def test_precision_recall_f1_score_multilabel_2():
    # Test precision_recall_f1_score on a crafted multilabel example 2
    # Second crafted example
    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0]])
    y_pred = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 0, 0]])

    # tp = [ 0.  1.  0.  0.]
    # fp = [ 1.  0.  0.  2.]
    # fn = [ 1.  1.  1.  0.]

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average=None)
    assert_array_almost_equal(p, [0.0, 1.0, 0.0, 0.0], 2)
    assert_array_almost_equal(r, [0.0, 0.5, 0.0, 0.0], 2)
    assert_array_almost_equal(f, [0.0, 0.66, 0.0, 0.0], 2)
    assert_array_almost_equal(s, [1, 2, 1, 0], 2)

    f2 = fbeta_score(y_true, y_pred, beta=2, average=None)
    support = s
    assert_array_almost_equal(f2, [0, 0.55, 0, 0], 2)

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average="micro")
    assert_almost_equal(p, 0.25)
    assert_almost_equal(r, 0.25)
    assert_almost_equal(f, 2 * 0.25 * 0.25 / 0.5)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2,
                                    average="micro"),
                        (1 + 4) * p * r / (4 * p + r))

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average="macro")
    assert_almost_equal(p, 0.25)
    assert_almost_equal(r, 0.125)
    assert_almost_equal(f, 2 / 12)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2,
                                    average="macro"),
                        np.mean(f2))

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average="weighted")
    assert_almost_equal(p, 2 / 4)
    assert_almost_equal(r, 1 / 4)
    assert_almost_equal(f, 2 / 3 * 2 / 4)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2,
                                    average="weighted"),
                        np.average(f2, weights=support))

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average="samples")
    # Check samples
    # |h(x_i) inter y_i | = [0, 0, 1]
    # |y_i| = [1, 1, 2]
    # |h(x_i)| = [1, 1, 2]

    assert_almost_equal(p, 1 / 6)
    assert_almost_equal(r, 1 / 6)
    assert_almost_equal(f, 2 / 4 * 1 / 3)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2,
                                    average="samples"),
                        0.1666, 2)


@ignore_warnings
@pytest.mark.parametrize('zero_division', ["warn", 0, 1])
def test_precision_recall_f1_score_with_an_empty_prediction(zero_division):
    y_true = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0]])
    y_pred = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])

    # true_pos = [ 0.  1.  1.  0.]
    # false_pos = [ 0.  0.  0.  1.]
    # false_neg = [ 1.  1.  0.  0.]
    zero_division = 1.0 if zero_division == 1.0 else 0.0
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average=None,
                                                 zero_division=zero_division)
    assert_array_almost_equal(p, [zero_division, 1.0, 1.0, 0.0], 2)
    assert_array_almost_equal(r, [0.0, 0.5, 1.0, zero_division], 2)
    assert_array_almost_equal(f, [0.0, 1 / 1.5, 1, 0.0], 2)
    assert_array_almost_equal(s, [1, 2, 1, 0], 2)

    f2 = fbeta_score(y_true, y_pred, beta=2, average=None,
                     zero_division=zero_division)
    support = s
    assert_array_almost_equal(f2, [0, 0.55, 1, 0], 2)

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average="macro",
                                                 zero_division=zero_division)
    assert_almost_equal(p, (2 + zero_division) / 4)
    assert_almost_equal(r, (1.5 + zero_division) / 4)
    assert_almost_equal(f, 2.5 / (4 * 1.5))
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2,
                                    average="macro"),
                        np.mean(f2))

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average="micro",
                                                 zero_division=zero_division)
    assert_almost_equal(p, 2 / 3)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 2 / 3 / (2 / 3 + 0.5))
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2,
                                    average="micro",
                                    zero_division=zero_division),
                        (1 + 4) * p * r / (4 * p + r))

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average="weighted",
                                                 zero_division=zero_division)
    assert_almost_equal(p, 3 / 4 if zero_division == 0 else 1.0)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, (2 / 1.5 + 1) / 4)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2,
                                    average="weighted",
                                    zero_division=zero_division),
                        np.average(f2, weights=support),
                        )

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average="samples")
    # |h(x_i) inter y_i | = [0, 0, 2]
    # |y_i| = [1, 1, 2]
    # |h(x_i)| = [0, 1, 2]
    assert_almost_equal(p, 1 / 3)
    assert_almost_equal(r, 1 / 3)
    assert_almost_equal(f, 1 / 3)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2,
                                    average="samples",
                                    zero_division=zero_division),
                        0.333, 2)

# Test f1_score end




# Test recall_score  begin

@pytest.mark.parametrize('zero_division', ["warn", 0, 1])
def test_recall_warnings(zero_division):
    assert_no_warnings(recall_score,
                       np.array([[1, 1], [1, 1]]),
                       np.array([[0, 0], [0, 0]]),
                       average='micro', zero_division=zero_division)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        recall_score(np.array([[0, 0], [0, 0]]),
                     np.array([[1, 1], [1, 1]]),
                     average='micro', zero_division=zero_division)
        if zero_division == "warn":
            assert (str(record.pop().message) ==
                    'Recall is ill-defined and '
                    'being set to 0.0 due to no true samples.'
                    ' Use `zero_division` parameter to control'
                    ' this behavior.')
        else:
            assert len(record) == 0

        recall_score([0, 0], [0, 0])
        if zero_division == "warn":
            assert (str(record.pop().message) ==
                    'Recall is ill-defined and '
                    'being set to 0.0 due to no true samples.'
                    ' Use `zero_division` parameter to control'
                    ' this behavior.')


# @ignore_warnings
# def test_precision_recall_f_binary_single_class():
#     # Test precision, recall and F-scores behave with a single positive or
#     # negative class
#     # Such a case may occur with non-stratified cross-validation
#     assert 1. == precision_score([1, 1], [1, 1])
#     assert 1. == recall_score([1, 1], [1, 1])
#     assert 1. == f1_score([1, 1], [1, 1])
#     assert 1. == fbeta_score([1, 1], [1, 1], 0)
#
#     assert 0. == precision_score([-1, -1], [-1, -1])
#     assert 0. == recall_score([-1, -1], [-1, -1])
#     assert 0. == f1_score([-1, -1], [-1, -1])
#     assert 0. == fbeta_score([-1, -1], [-1, -1], float('inf'))
#     assert fbeta_score([-1, -1], [-1, -1], float('inf')) == pytest.approx(
#         fbeta_score([-1, -1], [-1, -1], beta=1e5))
# Test recall_score end










# Test  classification_report  begin

@pytest.mark.parametrize('zero_division', ["warn", 0, 1])
def test_classification_report_zero_division_warning(zero_division):
    y_true, y_pred = ["a", "b", "c"], ["a", "b", "d"]
    with warnings.catch_warnings(record=True) as record:
        classification_report(
            y_true, y_pred, zero_division=zero_division, output_dict=True)
        if zero_division == "warn":
            assert len(record) > 1
            for item in record:
                msg = ("Use `zero_division` parameter to control this "
                       "behavior.")
                assert msg in str(item.message)
        else:
            assert not record


# def test_classification_report_multiclass():
#     # Test performance report
#     iris = datasets.load_iris()
#     y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)
#
#     # print classification report with class names
#     expected_report = """\
#               precision    recall  f1-score   support
#
#       setosa       0.83      0.79      0.81        24
#   versicolor       0.33      0.10      0.15        31
#    virginica       0.42      0.90      0.57        20
#
#     accuracy                           0.53        75
#    macro avg       0.53      0.60      0.51        75
# weighted avg       0.51      0.53      0.47        75
# """
#     report = classification_report(
#         y_true, y_pred, labels=np.arange(len(iris.target_names)),
#         target_names=iris.target_names)
#     assert report == expected_report
#
#
# def test_classification_report_multiclass_balanced():
#     y_true, y_pred = [0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]
#
#     expected_report = """\
#               precision    recall  f1-score   support
#
#            0       0.33      0.33      0.33         3
#            1       0.33      0.33      0.33         3
#            2       0.33      0.33      0.33         3
#
#     accuracy                           0.33         9
#    macro avg       0.33      0.33      0.33         9
# weighted avg       0.33      0.33      0.33         9
# """
#     report = classification_report(y_true, y_pred)
#     assert report == expected_report
#
#
# def test_classification_report_multiclass_with_label_detection():
#     iris = datasets.load_iris()
#     y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)
#
#     # print classification report with label detection
#     expected_report = """\
#               precision    recall  f1-score   support
#
#            0       0.83      0.79      0.81        24
#            1       0.33      0.10      0.15        31
#            2       0.42      0.90      0.57        20
#
#     accuracy                           0.53        75
#    macro avg       0.53      0.60      0.51        75
# weighted avg       0.51      0.53      0.47        75
# """
#     report = classification_report(y_true, y_pred)
#     assert report == expected_report
#
#
# def test_classification_report_multiclass_with_digits():
#     # Test performance report with added digits in floating point values
#     iris = datasets.load_iris()
#     y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)
#
#     # print classification report with class names
#     expected_report = """\
#               precision    recall  f1-score   support
#
#       setosa    0.82609   0.79167   0.80851        24
#   versicolor    0.33333   0.09677   0.15000        31
#    virginica    0.41860   0.90000   0.57143        20
#
#     accuracy                        0.53333        75
#    macro avg    0.52601   0.59615   0.50998        75
# weighted avg    0.51375   0.53333   0.47310        75
# """
#     report = classification_report(
#         y_true, y_pred, labels=np.arange(len(iris.target_names)),
#         target_names=iris.target_names, digits=5)
#     assert report == expected_report
#
#
# def test_classification_report_multiclass_with_string_label():
#     y_true, y_pred, _ = make_prediction(binary=False)
#
#     y_true = np.array(["blue", "green", "red"])[y_true]
#     y_pred = np.array(["blue", "green", "red"])[y_pred]
#
#     expected_report = """\
#               precision    recall  f1-score   support
#
#         blue       0.83      0.79      0.81        24
#        green       0.33      0.10      0.15        31
#          red       0.42      0.90      0.57        20
#
#     accuracy                           0.53        75
#    macro avg       0.53      0.60      0.51        75
# weighted avg       0.51      0.53      0.47        75
# """
#     report = classification_report(y_true, y_pred)
#     assert report == expected_report
#
#     expected_report = """\
#               precision    recall  f1-score   support
#
#            a       0.83      0.79      0.81        24
#            b       0.33      0.10      0.15        31
#            c       0.42      0.90      0.57        20
#
#     accuracy                           0.53        75
#    macro avg       0.53      0.60      0.51        75
# weighted avg       0.51      0.53      0.47        75
# """
#     report = classification_report(y_true, y_pred,
#                                    target_names=["a", "b", "c"])
#     assert report == expected_report
#
#
# def test_classification_report_multiclass_with_unicode_label():
#     y_true, y_pred, _ = make_prediction(binary=False)
#
#     labels = np.array(["blue\xa2", "green\xa2", "red\xa2"])
#     y_true = labels[y_true]
#     y_pred = labels[y_pred]
#
#     expected_report = """\
#               precision    recall  f1-score   support
#
#        blue\xa2       0.83      0.79      0.81        24
#       green\xa2       0.33      0.10      0.15        31
#         red\xa2       0.42      0.90      0.57        20
#
#     accuracy                           0.53        75
#    macro avg       0.53      0.60      0.51        75
# weighted avg       0.51      0.53      0.47        75
# """
#     report = classification_report(y_true, y_pred)
#     assert report == expected_report
#
#
# def test_classification_report_multiclass_with_long_string_label():
#     y_true, y_pred, _ = make_prediction(binary=False)
#
#     labels = np.array(["blue", "green" * 5, "red"])
#     y_true = labels[y_true]
#     y_pred = labels[y_pred]
#
#     expected_report = """\
#                            precision    recall  f1-score   support
#
#                      blue       0.83      0.79      0.81        24
# greengreengreengreengreen       0.33      0.10      0.15        31
#                       red       0.42      0.90      0.57        20
#
#                  accuracy                           0.53        75
#                 macro avg       0.53      0.60      0.51        75
#              weighted avg       0.51      0.53      0.47        75
# """
#
#     report = classification_report(y_true, y_pred)
#     assert report == expected_report
#
#
# def test_classification_report_labels_target_names_unequal_length():
#     y_true = [0, 0, 2, 0, 0]
#     y_pred = [0, 2, 2, 0, 0]
#     target_names = ['class 0', 'class 1', 'class 2']
#
#     assert_warns_message(UserWarning,
#                          "labels size, 2, does not "
#                          "match size of target_names, 3",
#                          classification_report,
#                          y_true, y_pred, labels=[0, 2],
#                          target_names=target_names)
#
#
# def test_classification_report_no_labels_target_names_unequal_length():
#     y_true = [0, 0, 2, 0, 0]
#     y_pred = [0, 2, 2, 0, 0]
#     target_names = ['class 0', 'class 1', 'class 2']
#
#     err_msg = ("Number of classes, 2, does not "
#                "match size of target_names, 3. "
#                "Try specifying the labels parameter")
#     with pytest.raises(ValueError, match=err_msg):
#         classification_report(y_true, y_pred, target_names=target_names)
#
#
# @ignore_warnings
# def test_multilabel_classification_report():
#     n_classes = 4
#     n_samples = 50
#
#     _, y_true = make_multilabel_classification(n_features=1,
#                                                n_samples=n_samples,
#                                                n_classes=n_classes,
#                                                random_state=0)
#
#     _, y_pred = make_multilabel_classification(n_features=1,
#                                                n_samples=n_samples,
#                                                n_classes=n_classes,
#                                                random_state=1)
#
#     expected_report = """\
#               precision    recall  f1-score   support
#
#            0       0.50      0.67      0.57        24
#            1       0.51      0.74      0.61        27
#            2       0.29      0.08      0.12        26
#            3       0.52      0.56      0.54        27
#
#    micro avg       0.50      0.51      0.50       104
#    macro avg       0.45      0.51      0.46       104
# weighted avg       0.45      0.51      0.46       104
#  samples avg       0.46      0.42      0.40       104
# """
#
#     report = classification_report(y_true, y_pred)
#     assert report == expected_report

# Test  classification_report  end




# Test  precision_recall_fscore_support  begin
# @ignore_warnings
# def test_precision_recall_fscore_support_errors():
#     y_true, y_pred, _ = make_prediction(binary=True)
#
#     # Bad beta
#     with pytest.raises(ValueError):
#         precision_recall_fscore_support(y_true, y_pred, beta=-0.1)
#
#     # Bad pos_label
#     with pytest.raises(ValueError):
#         precision_recall_fscore_support(y_true, y_pred,
#                                         pos_label=2,
#                                         average='binary')
#
#     # Bad average option
#     with pytest.raises(ValueError):
#         precision_recall_fscore_support([0, 1, 2], [1, 2, 0],
#                                         average='mega')

def test_precision_recall_f_unused_pos_label():
    # Check warning that pos_label unused when set to non-default value
    # but average != 'binary'; even if data is binary.
    assert_warns_message(UserWarning,
                         "Note that pos_label (set to 2) is "
                         "ignored when average != 'binary' (got 'macro'). You "
                         "may use labels=[pos_label] to specify a single "
                         "positive class.", precision_recall_fscore_support,
                         [1, 2, 1], [1, 2, 2], pos_label=2, average='macro')

@pytest.mark.parametrize('average',
                         ['samples', 'micro', 'macro', 'weighted', None])
def test_precision_refcall_f1_score_multilabel_unordered_labels(average):
    # test that labels need not be sorted in the multilabel case
    y_true = np.array([[1, 1, 0, 0]])
    y_pred = np.array([[0, 0, 1, 1]])
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[3, 0, 1, 2], warn_for=[], average=average)
    assert_array_equal(p, 0)
    assert_array_equal(r, 0)
    assert_array_equal(f, 0)
    if average is None:
        assert_array_equal(s, [0, 1, 1, 0])

# Test  precision_recall_fscore_support  end



# Test  average_precision_score  begin

def test_average_precision_score_score_non_binary_class():
    # Test that average_precision_score function returns an error when trying
    # to compute average_precision_score for multiclass task.
    rng = check_random_state(404)
    y_pred = rng.rand(10)

    # y_true contains three different class values
    y_true = rng.randint(0, 3, size=10)
    err_msg = "multiclass format is not supported"
    with pytest.raises(ValueError, match=err_msg):
        average_precision_score(y_true, y_pred)


def test_average_precision_score_duplicate_values():
    # Duplicate values with precision-recall require a different
    # processing than when computing the AUC of a ROC, because the
    # precision-recall curve is a decreasing curve
    # The following situation corresponds to a perfect
    # test statistic, the average_precision_score should be 1
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    y_score = [0, .1, .1, .4, .5, .6, .6, .9, .9, 1, 1]
    assert average_precision_score(y_true, y_score) == 1


def test_average_precision_score_tied_values():
    # Here if we go from left to right in y_true, the 0 values are
    # are separated from the 1 values, so it appears that we've
    # Correctly sorted our classifications. But in fact the first two
    # values have the same score (0.5) and so the first two values
    # could be swapped around, creating an imperfect sorting. This
    # imperfection should come through in the end score, making it less
    # than one.
    y_true = [0, 1, 1]
    y_score = [.5, .5, .6]
    assert average_precision_score(y_true, y_score) != 1.

# Test  average_precision_score  end




