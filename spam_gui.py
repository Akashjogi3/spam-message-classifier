
import tkinter as tk
from tkinter import messagebox

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# STEP 1: LOAD THE DATASET
# --------------------------------------------------

try:
    data = pd.read_csv(
        "data/SMSSpamCollection",
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    data = data.dropna(subset=["label", "message"])
    data = data.drop_duplicates()

    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str).str.strip()

    data = data[
        data["label"].isin(["ham", "spam"])
        & (data["message"] != "")
    ]

    if data.empty or data["label"].nunique() < 2:
        raise ValueError(
            "The dataset must contain both ham and spam messages."
        )

except Exception as error:
    root = tk.Tk()
    root.withdraw()

    messagebox.showerror(
        "Dataset Error",
        "Could not load the dataset.\n\n"
        "Please check that this file exists:\n"
        "data/SMSSpamCollection\n\n"
        f"Details: {error}"
    )

    root.destroy()
    raise SystemExit


# --------------------------------------------------
# STEP 2: PREPARE TRAINING AND TEST DATA
# --------------------------------------------------

messages = data["message"]
labels = data["label"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


# --------------------------------------------------
# STEP 3: CONVERT TEXT INTO NUMBERS
# --------------------------------------------------

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# --------------------------------------------------
# STEP 4: TRAIN THE MACHINE LEARNING MODEL
# --------------------------------------------------

model = MultinomialNB()
model.fit(X_train, y_train)


# --------------------------------------------------
# STEP 5: EVALUATE MODEL ACCURACY
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Spam Message Classifier")
print("-" * 35)
print(f"Total messages: {len(data)}")
print(f"Training messages: {len(X_train_text)}")
print(f"Testing messages: {len(X_test_text)}")
print(f"Test accuracy: {accuracy:.2%}")


# --------------------------------------------------
# STEP 6: CHECK A MESSAGE
# --------------------------------------------------

def check_message():
    message = message_box.get("1.0", tk.END).strip()

    if not message:
        messagebox.showwarning(
            "Empty Message",
            "Please enter a message before checking."
        )
        return

    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]

    if prediction == "spam":
        result_label.config(
            text="SPAM MESSAGE",
            fg="#D93025"
        )

        detail_label.config(
            text="Warning: This message may be unwanted or fraudulent.",
            fg="#B3261E"
        )

    else:
        result_label.config(
            text="NOT SPAM",
            fg="#188038"
        )

        detail_label.config(
            text="The model classifies this as a normal message.",
            fg="#137333"
        )


# --------------------------------------------------
# STEP 7: CLEAR THE MESSAGE
# --------------------------------------------------

def clear_message():
    message_box.delete("1.0", tk.END)

    result_label.config(
        text="Result will appear here",
        fg="#333333"
    )

    detail_label.config(text="")


# --------------------------------------------------
# STEP 8: CREATE THE GUI WINDOW
# --------------------------------------------------

root = tk.Tk()

root.title("Spam Message Classifier")
root.geometry("620x520")
root.resizable(False, False)
root.configure(bg="#F3F6FA")


# Heading
heading = tk.Label(
    root,
    text="Spam Message Classifier",
    font=("Arial", 22, "bold"),
    bg="#F3F6FA",
    fg="#183153"
)
heading.pack(pady=(25, 8))


# Subtitle
subtitle = tk.Label(
    root,
    text="Enter a message to check whether it is spam",
    font=("Arial", 11),
    bg="#F3F6FA",
    fg="#555555"
)
subtitle.pack(pady=(0, 10))


# Accuracy display
accuracy_label = tk.Label(
    root,
    text=f"Model Test Accuracy: {accuracy:.2%}",
    font=("Arial", 11, "bold"),
    bg="#F3F6FA",
    fg="#2457C5"
)
accuracy_label.pack(pady=(0, 18))


# Message input label
input_label = tk.Label(
    root,
    text="Your message:",
    font=("Arial", 12, "bold"),
    bg="#F3F6FA",
    fg="#222222"
)
input_label.pack(anchor="w", padx=45)


# Message input box
message_box = tk.Text(
    root,
    height=5,
    width=60,
    font=("Arial", 11),
    wrap="word",
    bd=2,
    relief="groove"
)
message_box.pack(padx=40, pady=10)


# Buttons container
button_frame = tk.Frame(
    root,
    bg="#F3F6FA"
)
button_frame.pack(pady=10)


# Check Message button
check_button = tk.Button(
    button_frame,
    text="Check Message",
    command=check_message,
    font=("Arial", 11, "bold"),
    bg="#2457C5",
    fg="white",
    activebackground="#1745A5",
    activeforeground="white",
    width=16,
    cursor="hand2"
)
check_button.grid(row=0, column=0, padx=8)


# Clear button
clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_message,
    font=("Arial", 11),
    width=10,
    cursor="hand2"
)
clear_button.grid(row=0, column=1, padx=8)


# Prediction result
result_label = tk.Label(
    root,
    text="Result will appear here",
    font=("Arial", 18, "bold"),
    bg="#F3F6FA",
    fg="#333333"
)
result_label.pack(pady=(18, 5))


# Additional result information
detail_label = tk.Label(
    root,
    text="",
    font=("Arial", 10),
    bg="#F3F6FA",
    fg="#555555",
    wraplength=520
)
detail_label.pack(pady=5)


# Footer
footer = tk.Label(
    root,
    text=(
        f"Model: Naive Bayes | "
        f"Training messages: {len(X_train_text):,}"
    ),
    font=("Arial", 9),
    bg="#F3F6FA",
    fg="#777777"
)
footer.pack(side="bottom", pady=15)


# --------------------------------------------------
# STEP 9: START THE APPLICATION
# --------------------------------------------------

root.mainloop()
