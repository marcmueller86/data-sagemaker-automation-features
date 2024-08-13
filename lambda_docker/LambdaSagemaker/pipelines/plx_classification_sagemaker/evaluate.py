"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle
import tarfile
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import xgboost


from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, auc, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix, roc_curve, matthews_corrcoef
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def process_predictions_dataframe(df, percentiles=None):
    """
    Process the predictions dataframe and return a DataFrame with calculated statistics.

    Parameters:
    df (DataFrame): DataFrame with columns 'y_test' and 'y_pred_proba'.
    percentiles (list): List of percentiles to be calculated. Default is a predefined list.

    Returns:
    DataFrame: DataFrame with calculated statistics.
    """
    if percentiles is None:
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 98]

    # Sort the data by y_pred_proba in descending order
    sorted_df = df.sort_values(by='y_pred_proba', ascending=False).reset_index(drop=True)

    # Calculate the total number of positive labels
    total_positives = sorted_df['y_test'].sum()

    # Function to calculate the coverage
    def calculate_coverage(percentage, data, total_positives):
        required_positives = total_positives * percentage / 100
        cumulative_positives = data['y_test'].cumsum()
        row_threshold = cumulative_positives.searchsorted(required_positives, side='right')
        return row_threshold + 1

    # Calculate coverage rows and rows for percentiles
    coverage_rows = {perc: calculate_coverage(perc, sorted_df, total_positives) for perc in percentiles}
    rows_for_percentiles = {perc: int(len(sorted_df) * perc / 100) for perc in percentiles}

    # Initialize dictionaries for storing statistics
    stats = {
        "positive_labels_captured": {},
        "percentage_of_emails_used": {},
        "percentage_of_positives_covered": {},
        "mean_probability": {},
        "median_probability": {}
    }

    # Calculate statistics for each percentile chunk
    for perc in percentiles:
        row_count = rows_for_percentiles[perc]
        data_slice = sorted_df.iloc[:row_count]

        stats["positive_labels_captured"][perc] = data_slice['y_test'].sum()
        stats["percentage_of_emails_used"][perc] = 100 * row_count / len(sorted_df)
        stats["percentage_of_positives_covered"][perc] = 100 * stats["positive_labels_captured"][perc] / total_positives
        stats["mean_probability"][perc] = data_slice['y_pred_proba'].mean()
        stats["median_probability"][perc] = data_slice['y_pred_proba'].median()

    # Create the DataFrame
    results_df = pd.DataFrame({
        "Percentile": percentiles,
        "Rows (Emails to Send)": [rows_for_percentiles[perc] for perc in percentiles],
        "Positive Labels Captured": [stats["positive_labels_captured"][perc] for perc in percentiles],
        "% of Overall Emails Used": [stats["percentage_of_emails_used"][perc] for perc in percentiles],
        "% of Positives Covered": [stats["percentage_of_positives_covered"][perc] for perc in percentiles],
        "Mean Probability": [stats["mean_probability"][perc] for perc in percentiles],
        "Median Probability": [stats["median_probability"][perc] for perc in percentiles]
    })

    # Convert to German number format
    results_df['Rows (Emails to Send)'] = results_df['Rows (Emails to Send)'].apply(lambda x: "{:,.0f}".format(x).replace(',', '.'))
    results_df['% of Overall Emails Used'] = results_df['% of Overall Emails Used'].apply(lambda x: "{:.2f}".format(x).replace('.', ','))
    results_df['% of Positives Covered'] = results_df['% of Positives Covered'].apply(lambda x: "{:.2f}".format(x).replace('.', ','))
    results_df['Mean Probability'] = results_df['Mean Probability'].apply(lambda x: "{:.4f}".format(x).replace('.', ','))
    results_df['Median Probability'] = results_df['Median Probability'].apply(lambda x: "{:.4f}".format(x).replace('.', ','))

    return results_df

# Example usage:
# df = pd.DataFrame({'y_test': y_test, 'y_pred_proba': y_pred_proba})
# result_df = process_predictions_dataframe(df)
# result_df.to_csv('output_statistics.csv', index=False)


def evaluate_model(y_test, y_pred_proba, threshold):
# Basic threshold-based metrics
    y_pred = np.where(y_pred_proba >= threshold, 1, 0)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)
    mcc = matthews_corrcoef(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    TP, FP, FN, TN = cm.ravel()

    # Sorting and calculating percentiles
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_test_sorted = y_test[sorted_indices]
    total_positives = np.sum(y_test_sorted)
    
    results = {
        "threshold": float(threshold),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "mcc": float(mcc),
        "balanced_accuracy": float(balanced_acc),
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "TN": int(TN),
    }

    # Adding percentile coverages to the results
    for percentile in range(10, 101, 10):
        index = int(len(y_test_sorted) * percentile / 100)
        results[f'percentile_coverage_{percentile}'] = sum(y_test_sorted[:index]) / total_positives
    return results


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("/opt/ml/processing/evaluation/roc_curve.png")
    plt.close()

def plot_feature_importance(model):
    feature_names = []
    try:
        with open('/opt/ml/processing/feature/feature_names.json', 'r') as file:
            feature_names = json.load(file)

        model.get_booster().feature_names = feature_names
    except:
        logger.info("No feature names found.")
    
    plt.figure(figsize=(10, 8))
    xgboost.plot_importance(model)
    plt.savefig("/opt/ml/processing/evaluation/feature_importance.png", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    feature_name = "/opt/ml/processing/feature/feature_names.json"
    test_set_mapping = "/opt/ml/processing/data/test_customer_campaign_mapping.csv"

    logger.debug("Loading data")
    df = pd.read_csv(test_path)
    logger.debug(f"df columns: {df.columns}")

    df = pd.read_csv(test_path, header=None)
    logger.info(df.columns)
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)

    y_pred_proba = model.predict(X_test)
    
    # Calculate class balance
    class_counts = df.iloc[:, 0].value_counts(normalize=True).to_dict()

    # Evaluate model for different thresholds
    metrics = [
        evaluate_model(y_test, y_pred_proba, 0.5),
    ]

    # Organize into a structured report
    report_dict = {
        "metrics": metrics,
        "regression_metrics": {
             "f1": metrics[0]["f1"],
             "accuracy": metrics[0]["accuracy"],
         }    
         }


    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

        # Create a DataFrame from y_test and y_pred_proba
    df = pd.DataFrame({
        'y_test': y_test,
        'y_pred_proba': y_pred_proba
    })

    # Write the DataFrame to a CSV file
    evaluation_path = f"{output_dir}/target_probabilities.csv"

    df.to_csv(evaluation_path, index=False)


    # Plotting ROC curve
    plot_roc_curve(y_test, y_pred_proba)
    # Plotting Feature Importance
    plot_feature_importance(model)

    with open(feature_name) as fp:
        feature_names = json.load(fp)
    with open(f"{output_dir}/feature_names.json", 'w') as file:
        file.write(feature_names)
        
    
    test_customer_campaign_mapping = pd.read_csv(test_set_mapping)
    test_customer_campaign_mapping.to_csv(f"{output_dir}/test_set_mapping.csv", index=False)

    result_df = process_predictions_dataframe(df)
    result_df.to_csv('output_statistics.csv', index=False)
