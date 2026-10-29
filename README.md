
# Spam Message Classifier

## Project Overview
The Spam Message Classifier is a Python-based machine learning application that identifies whether a text message is spam or not spam. It provides a simple graphical user interface built using Tkinter.

## Objectives
- Identify spam-like and normal messages.
- Apply machine learning to text classification.
- Provide a simple and user-friendly interface.
- Evaluate the model using a separate test dataset.

## Technologies Used
- Python
- Tkinter
- Pandas
- Scikit-learn
- CountVectorizer
- Multinomial Naive Bayes
- UCI SMS Spam Collection dataset

## Features
- Graphical user interface.
- Message input and classification.
- Spam and not-spam result display.
- Model test accuracy display.
- Clear button to reset the input.
- Dataset-based model training and evaluation.

## Project Structure

```text
spam_email_classifier/
├── data/
│   └── SMSSpamCollection
├── spam_classifier.py
├── spam_gui.py
├── README.md
├── PROJECT_REPORT.md
└── requirements.txt
```

## Installation and Execution

### 1. Install Python
Install Python 3 if it is not already installed.

### 2. Install the required libraries
Open the terminal in the project directory and run:

```powershell
.\.venv\Scripts\python.exe -m pip install pandas scikit-learn
```

### 3. Prepare the dataset
Download the UCI SMS Spam Collection dataset and place the extracted `SMSSpamCollection` file inside the `data` folder.

Dataset source:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

### 4. Run the application

```powershell
.\.venv\Scripts\python.exe spam_gui.py
```

## Model
The application uses:
- CountVectorizer to convert text into numerical features.
- Multinomial Naive Bayes to classify messages.
- An 80/20 train-test split to evaluate performance.

## Results
In the current test run, the model achieved 98.16% accuracy. Actual performance may vary with different data and messages.

## Limitations
This version uses SMS messages rather than full email messages. The model may incorrectly classify some messages and should not be treated as a guarantee that a message is safe.

## Future Enhancements
- Train and evaluate on a genuine email dataset.
- Add precision, recall, and F1-score reporting.
- Improve text preprocessing.
- Save the trained model for faster startup.
- Add a more advanced interface.

## Author
MCA Student
