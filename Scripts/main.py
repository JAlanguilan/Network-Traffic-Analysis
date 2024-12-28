import os
import glob
import pandas as pd
import json
import logging
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.metrics import classification_report, confusion_matrix
from Scripts.helpers import load_data, clean_data, split_data
from Scripts.multiclass_classification import (
    direct_multiclass_train,
    direct_multiclass_test,
    data_resampling,
    improved_data_split,
    get_binary_dataset,
)

# Configuration Parameters
FOLDER_PATH = 'MachineLearningCVE'
TARGET_COLUMN = 'Label'
SAMPLING_STRATEGY = 'auto'
MODELS = ['mlp', 'rf']
LOG_FILE = "process.log"
RESULTS_FILE = "results.json"
PDF_FILE = "results.pdf"

# Setup Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# Store results for PDF
all_results = []


def save_results(file_name, model_name, task_type, metrics, y_true, y_pred):
    """
    Save results into a JSON file and store them in a global list for PDF output.
    """
    results = {
        "file": file_name,
        "task": task_type,
        "model": model_name.upper(),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # Save results to JSON file
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            json.dump([], f)

    with open(RESULTS_FILE, "r+") as f:
        data = json.load(f)
        data.append(results)
        f.seek(0)
        json.dump(data, f, indent=4)

    # Add results to the global list for PDF
    all_results.append(results)


def write_results_to_pdf():
    """
    Write all results stored in 'all_results' to a PDF file.
    """
    c = canvas.Canvas(PDF_FILE, pagesize=letter)
    width, height = letter

    # PDF Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, height - 40, "Classification Results Report")

    # Write each result to the PDF
    y = height - 70
    for result in all_results:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(30, y, f"File: {result['file']}, Task: {result['task']}, Model: {result['model']}")
        y -= 20

        c.setFont("Helvetica", 10)
        c.drawString(30, y, f"Accuracy: {result['accuracy']}")
        y -= 15
        c.drawString(30, y, f"Precision: {result['precision']}")
        y -= 15
        c.drawString(30, y, f"Recall: {result['recall']}")
        y -= 15
        c.drawString(30, y, f"F1-Score: {result['f1_score']}")
        y -= 20

        c.drawString(30, y, "Confusion Matrix:")
        y -= 15
        for row in result["confusion_matrix"]:
            c.drawString(50, y, str(row))
            y -= 15

        y -= 10
        if y < 50:  # Add a new page if content overflows
            c.showPage()
            y = height - 40

    c.save()
    print(f"Results written to {PDF_FILE}")


def process_file(csv_file):
    """
    Process a single CSV file: load, clean, and perform classification tasks.
    """
    try:
        logger.info(f"Processing file: {csv_file}")
        df = load_data(csv_file, target_col=TARGET_COLUMN)
        df = clean_data(df, target_col=TARGET_COLUMN)

        # Direct Multi-Class Classification
        run_multiclass_classification(df, csv_file)

        # Resampled Multi-Class Classification
        run_resampled_multiclass_classification(df, csv_file)

        # Hierarchical Binary Classification
        run_hierarchical_classification(df, csv_file)

    except Exception as e:
        logger.error(f"Error processing file {csv_file}: {str(e)}")


def run_multiclass_classification(df, file_name):
    logger.info("Running Direct Multi-Class Classification")
    X_train, X_test, y_train, y_test = split_data(df, TARGET_COLUMN)

    for model_name in MODELS:
        model = direct_multiclass_train(model_name, X_train, y_train)
        metrics = direct_multiclass_test(model, X_test, y_test)
        save_results(file_name, model_name, "Multi-Class", metrics, y_test, model.predict(X_test))


def run_resampled_multiclass_classification(df, file_name):
    logger.info("Running Resampled Multi-Class Classification")
    df_resampled = data_resampling(df, TARGET_COLUMN, SAMPLING_STRATEGY)
    X_train, X_test, y_train, y_test = split_data(df_resampled, TARGET_COLUMN)

    for model_name in MODELS:
        model = direct_multiclass_train(model_name, X_train, y_train)
        metrics = direct_multiclass_test(model, X_test, y_test)
        save_results(file_name, model_name, "Resampled Multi-Class", metrics, y_test, model.predict(X_test))


def run_hierarchical_classification(df, file_name):
    logger.info("Running Hierarchical Classification")
    df_train, df_test = improved_data_split(df, TARGET_COLUMN)
    df_train_binary = get_binary_dataset(df_train, TARGET_COLUMN)
    df_test_binary = get_binary_dataset(df_test, TARGET_COLUMN)

    df_train_resampled = data_resampling(df_train_binary, TARGET_COLUMN, SAMPLING_STRATEGY)

    X_train_bin = df_train_resampled.drop(TARGET_COLUMN, axis=1)
    y_train_bin = df_train_resampled[TARGET_COLUMN]
    X_test_bin = df_test_binary.drop(TARGET_COLUMN, axis=1)
    y_test_bin = df_test_binary[TARGET_COLUMN]

    for model_name in MODELS:
        model = direct_multiclass_train(model_name, X_train_bin, y_train_bin)
        metrics = direct_multiclass_test(model, X_test_bin, y_test_bin)
        save_results(file_name, model_name, "Binary Classification", metrics, y_test_bin, model.predict(X_test_bin))


def main():
    csv_files = glob.glob(os.path.join(FOLDER_PATH, '*.csv'))
    logger.info(f"Found {len(csv_files)} CSV files in folder '{FOLDER_PATH}'.")

    for csv_file in csv_files:
        process_file(csv_file)

    # Write results to PDF at the end
    write_results_to_pdf()
    logger.info("Processing complete. Results saved to results.json and results.pdf.")


if __name__ == "__main__":
    main()
