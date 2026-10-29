
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# STEP 1: Load the larger dataset
data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# Remove empty messages and duplicate rows
data = data.dropna(subset=["message", "label"])
data = data.drop_duplicates()
data["label"] = data["label"].str.strip().str.lower()

# Keep only valid labels
data = data[data["label"].isin(["ham", "spam"])]

print("Dataset loaded successfully!")
print("Total messages:", len(data))
print("\nLabel counts:")
print(data["label"].value_counts())

# STEP 2: Separate messages and labels
messages = data["message"].astype(str)
labels = data["label"]

# STEP 3: Split text before learning vocabulary
X_train_text, X_test_text, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# STEP 4: Convert text into numerical features
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# STEP 5: Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

print("\nModel training completed!")

# STEP 6: Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        target_names=["Not Spam", "Spam"],
        zero_division=0
    )
)

print("Confusion Matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

# STEP 7: Test new messages
print("\n--- Spam Message Classifier ---")
print("Type 'exit' to close the program.")

while True:
    message = input("\nEnter a message: ").strip()

    if message.lower() == "exit":
        print("Thank you for using the classifier!")
        break

    if not message:
        print("Please enter a message.")
        continue

    message_features = vectorizer.transform([message])
    result = model.predict(message_features)[0]

    if result == "spam":
        print("Prediction: SPAM")
    else:
        print("Prediction: NOT SPAM")
