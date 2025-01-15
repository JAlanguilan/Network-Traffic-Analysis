# Network-Traffic-Analysis

Multi-Class and Hierarchical Classification for Network Traffic Data

Description
-----------
This project is designed to handle multi-class classification and hierarchical binary classification tasks for Machine Learning-based datasets. 
It leverages Python libraries like Pandas, Scikit-learn, and Imbalanced-learn to:
    1. Load and Clean Data: Strip unnecessary columns, clean NaN values, and ensure usable formats.
    2. Multi-Class Classification: Train and evaluate MLP and Random Forest classifiers.
    3. Data Resampling: Address class imbalance using Random Under-Sampling.
    4. Hierarchical Classification:
        - Binary Classification (Benign vs. Malicious).
        - Malicious Multi-Class Classification


Project Structure
-----------------
.
|-- MachineLearningCVE/         # Folder containing CSV files

|-- helpers.py                  # Data loading, cleaning, and splitting utilities

|-- multiclass_classification.py # Multi-class training, testing, and resampling

|-- main.py                     # Main script to process all tasks

|-- process.log                 # Log file generated during execution

|-- results.json                # Evaluation results (accuracy, metrics, etc.)

|-- results.pdf                 # Formatted PDF report containing all results

|-- README.txt                  # Project documentation


Dependencies
------------
Python (>=3.7)
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
reportlab (for pdf generation)

You can install them using:
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn


How to use
-----------
Place data files
    - Add your CSV files to the folder MachineLearningCVE. The MachineLearningCVE folder should be in same directory as the scripts.

Run the Script: Execute the main script:
    - python main.py

- The script will process all CSV files in the folder.
- Logs will be saved to process.log.
- Results will be saved to results.json.

Review Results:
    - Classification results (accuracy, precision, recall, F1-score) will be printed to the console.
    - Summarized results will be logged and saved into results.json.
    - A formatted PDF report summarizing all evaluation results



Tasks Performed
---------------
1. Direct Multi-Class Classification:
    - Trains MLPClassifier and RandomForestClassifier.
    - Prints accuracy, classification report, and confusion matrix.
2. Multi-Class Classification with Resampling:
    - Balances the dataset using Random Under-Sampling.
    - Re-trains and evaluates models.
3. Hierarchical Classification:
    - Binary Classification: Distinguishes "Benign" vs. "Malicious".
    - Multi-Class Malicious: Performs classification only on malicious samples.


Outputs
-------
1. Console Outputs:
    - Accuracy, confusion matrices, and detailed classification reports for each task and model.
2. Log file process.log
    - Contains a records of all processed files, tasks, and evaluation results.
3. Results: results.json and results.pdf
    - Structured JSON and PDF file containing accuracy, classification metrics, and confusion matrices.
    - Note: results.json will contain evaluation metrics for all tasks (Direct Multi-Class, Resampled Multi-Class, Binary, and Malicious Multi-Class). Results for each model (MLP, RF) are appended sequentially.


Notes
-----
- Input CSV files must include a "Label" column for target classification.
- Missing values (NaN/Inf) are replaced with column means.
- Random seed is set for reproducibility.
