import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Mapping for months
    month_mapping = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }

    evidence = []  # List to store evidence data
    labels = []    # List to store label data
    
    # Function to convert each row of the CSV into the correct format
    def convert_row(row):
        return [
            int(row[0]), float(row[1]), int(row[2]), float(row[3]), int(row[4]),
            float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]),
            month_mapping[row[10]], int(row[11]), int(row[12]), int(row[13]), int(row[14]),
            1 if row[15] == "Returning_Visitor" else 0, 1 if row[16] == "TRUE" else 0
        ]
    
    # Read the csv file
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader) # Skips the header row
        for row in reader:
            # Convert and append each row to the evidence list
            evidence.append(convert_row(row))
            # Append the label (1 if TRUE, 0 if FALSE) to the labels list
            labels.append(1 if row[17] == "TRUE" else 0)

    return (evidence, labels)   # Return the tuple of evidence and labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    
    model = KNeighborsClassifier(n_neighbors=1)  # Initialize k-NN classifier with k=1
    model.fit(evidence, labels)  # Train the model on the provided evidence and labels
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    
    # sensitivity = (Predicted and labeled positive)/ (labeled positive)
    # Specificity = (predicted and labeled negative)/(labeled negative)
    # Initialize counters
    true_positives, true_negatives, total_positives, total_negatives = 0,0,0,0
    
    # Iterate over the actual and predicted labels
    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            total_positives += 1
            if actual == predicted:
                true_positives += 1
        else:
            total_negatives += 1
            if actual == predicted:
                true_negatives += 1
    # Calculate sensitivity (true positive rate)
    sensitivity = (true_positives / total_positives) if total_positives else 0
    # Calculate specificity (true negative rate)
    specificity = (true_negatives / total_negatives) if total_negatives else 0

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
