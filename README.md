
# Spam Mail Detector Using Machine Learning

## Objective
Build a classifier that identifies spam and non-spam (ham) messages using text data.

## Dataset
SMS Spam Collection dataset.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Methodology
1. Load and explore the dataset.
2. Check spam and ham message distribution.
3. Split data into training and testing sets.
4. Convert text into numerical features using TF-IDF.
5. Train a Multinomial Naive Bayes classifier.
6. Evaluate using accuracy, precision, recall, F1-score, and a confusion matrix.
7. Test the model on sample messages.

## How to Run
Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the project:

```bash
python spam_detector.py
```

## Results
Run the script and record the actual accuracy and evaluation metrics here.

## Conclusion
This project demonstrates basic natural language processing and machine learning classification to distinguish spam messages from legitimate messages.