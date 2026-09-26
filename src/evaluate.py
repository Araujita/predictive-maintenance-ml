import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# --- Cost assumptions (adjust based on the real factory context) ---
#
# FALSE_POSITIVE_COST: cost of a preventive inspection/maintenance done
#   unnecessarily (labor + short machine downtime).
#
# FALSE_NEGATIVE_COST: cost of a failure not detected in time — unplanned
#   downtime, emergency repair, lost production, and possible additional
#   damage to the equipment. Typically much higher than the cost of a
#   preventive inspection.
#
# TRUE_POSITIVE_BENEFIT: savings obtained by avoiding the failure through
#   planned preventive maintenance, already net of the cost of that
#   maintenance (i.e. it's a net value).
#
# These numbers are illustrative estimates based on typical industrial
# maintenance scenarios in SMEs; in a real case, they would come from the
# factory's own maintenance data (average downtime cost per hour, parts
# cost, emergency labor cost, etc.).
FALSE_POSITIVE_COST = -500
FALSE_NEGATIVE_COST = -15000
TRUE_POSITIVE_BENEFIT = 14000


def calculate_single_roi(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (
        fp * FALSE_POSITIVE_COST
        + fn * FALSE_NEGATIVE_COST
        + tp * TRUE_POSITIVE_BENEFIT
    )


def find_best_threshold(model, X_test, y_test):
    """
    Searches for the decision threshold that maximizes financial ROI,
    instead of using the default 0.5 threshold, which optimizes for
    statistical metrics (accuracy/F1) rather than real economic impact.

    Also returns the full history, useful for plotting ROI vs. threshold.
    """
    probabilities = model.predict_proba(X_test)[:, 1]

    best_threshold = 0.5
    max_profit = -float('inf')
    history = []

    for threshold in np.arange(0.01, 1.0, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        profit = calculate_single_roi(y_test, predictions)
        history.append((threshold, profit))

        if profit > max_profit:
            max_profit = profit
            best_threshold = threshold

    history_df = pd.DataFrame(history, columns=['threshold', 'profit'])
    return best_threshold, max_profit, history_df


def calculate_default_threshold_roi(model, X_test, y_test):
    """
    ROI using the default threshold (0.5), to serve as a baseline and
    quantify the gain obtained purely from optimizing the decision
    threshold.
    """
    predictions = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    return calculate_single_roi(y_test, predictions)
