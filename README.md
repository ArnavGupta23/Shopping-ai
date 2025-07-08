
# 🛒 Shopping AI - Predictive Model for Online Shopper Intention

This project uses a K-Nearest Neighbors (KNN) classifier to predict whether an online shopper will generate revenue based on their session behavior.

## 📁 Project Structure

```
Shopping-ai-main/
│
├── shopping.py         # Main Python script for loading data, training, and evaluating the model
├── shopping.csv        # Dataset containing session-level data from online shopping
└── .gitattributes      # Git settings file (optional)
```

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+shoppers+purchasing+intention+dataset)
- **Description**: The dataset includes features for each online shopping session and a label indicating whether the session generated revenue.

### Key Features
- Administrative, Informational, ProductRelated
- BounceRates, ExitRates
- PageValues, SpecialDay
- OperatingSystems, Browser, Region, TrafficType
- VisitorType (e.g., Returning_Visitor), Weekend (True/False)
- Revenue (target variable)

## 🚀 How to Run

### Step 1: Install Dependencies

Make sure you have Python and scikit-learn installed.

```bash
pip install scikit-learn
```

### Step 2: Run the Script

```bash
python shopping.py shopping.csv
```

This command loads the data, splits it into training and testing sets, trains a KNN model, and prints evaluation metrics.

## 🤖 Model Overview

- **Algorithm**: K-Nearest Neighbors (K=1)
- **Data Split**: 60% training, 40% testing
- **Metrics**:
  - Accuracy (correct vs incorrect predictions)
  - Sensitivity (True Positive Rate)
  - Specificity (True Negative Rate)

## 🧠 Functions

### `load_data(filename)`
Loads and preprocesses the shopping dataset from a CSV file.

### `train_model(evidence, labels)`
Trains a K=1 KNN classifier using the provided evidence and labels.

### `evaluate(labels, predictions)`
Computes sensitivity and specificity for the prediction results.

## ✅ Example Output

```
Correct: 3125
Incorrect: 375
True Positive Rate: 82.14%
True Negative Rate: 91.48%
```


