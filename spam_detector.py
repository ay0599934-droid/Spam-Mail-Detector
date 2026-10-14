# =====================================================
# PROJECT: SPAM MAIL DETECTOR USING MACHINE LEARNING
# Algorithm: Multinomial Naive Bayes
# Feature Extraction: TF-IDF
# Dataset: UCI SMS Spam Collection
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import zipfile
import io

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# STEP 1: LOAD DATASET

print("=" * 50)
print("SPAM MAIL DETECTOR")
print("=" * 50)

print("\nLoading SMS Spam Collection dataset...")

url = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms+spam+collection.zip"
)

try:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        zip_content = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
        with zip_file.open("SMSSpamCollection") as file:
            df = pd.read_csv(
                file,
                sep="\t",
                header=None,
                names=["label", "message"],
                encoding="utf-8"
            )

    print("Dataset loaded successfully!")

except Exception as error:
    print("\nDataset could not be downloaded.")
    print("Error:", error)
    print("Check your internet connection and try again.")
    raise SystemExit(1)


# STEP 2: EXPLORE DATASET

print("\nFirst 5 messages:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["label"].value_counts())


# STEP 3: DATA VISUALIZATION

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="label"
)

plt.title("Spam vs Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")
plt.tight_layout()
plt.show()


# STEP 4: PREPARE FEATURES AND LABELS

X = df["message"].astype(str)
y = df["label"].astype(str)

print("\nTotal messages:", len(X))


# STEP 5: TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))


# STEP 6: TF-IDF AND NAIVE BAYES MODEL

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])


# STEP 7: TRAIN MODEL

print("\nTraining the machine learning model...")

model.fit(X_train, y_train)

print("Model trained successfully!")


# STEP 8: MAKE PREDICTIONS

y_pred = model.predict(X_test)


# STEP 9: EVALUATE MODEL

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL EVALUATION RESULTS")
print("=" * 50)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["ham", "spam"],
        target_names=["Ham (Non-Spam)", "Spam"],
        zero_division=0
    )
)


# STEP 10: CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["ham", "spam"]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.title("Spam Detector Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()


# STEP 11: TEST CUSTOM MESSAGES

print("\n" + "=" * 50)
print("CUSTOM MESSAGE PREDICTIONS")
print("=" * 50)

sample_messages = [
    "Congratulations! You have won a free prize. Claim now!",
    "Hi, are we meeting for lunch today?",
    "URGENT! You have won a cash reward. Call now!",
    "Please send me the assignment notes."
]

for message in sample_messages:

    prediction = model.predict([message])[0]

    print("\nMessage:", message)
    print("Prediction:", prediction.upper())


# STEP 12: FINAL SUMMARY

print("\n" + "=" * 50)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 50)

print(f"Dataset size: {len(df)} messages")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Model accuracy: {accuracy * 100:.2f}%")

print("\nAlgorithm: Multinomial Naive Bayes")
print("Feature extraction: TF-IDF")
print("Classes: Spam and Ham")