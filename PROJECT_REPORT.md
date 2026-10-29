
# PROJECT REPORT
## Spam Message Classifier Using Machine Learning

### 1. Abstract
The Spam Message Classifier is a machine learning application developed using Python. It classifies text messages into two categories: spam and not spam. The application uses the CountVectorizer technique to convert text into numerical features and the Multinomial Naive Bayes algorithm to make predictions. A graphical user interface built with Tkinter allows users to enter messages and view the predicted result. The model is evaluated using a separate test dataset.

### 2. Introduction
Spam messages are unwanted messages that may contain misleading offers, advertisements, or suspicious links. Identifying these messages manually can be difficult. Machine learning can help recognize patterns in text and classify messages automatically.

This project demonstrates a basic text classification system with a simple interface for users.

### 3. Objectives
- Develop a basic machine learning text classifier.
- Classify messages as spam or not spam.
- Build an interactive graphical user interface.
- Evaluate model performance on unseen test data.
- Understand the application of machine learning in text classification.

### 4. Software and Technologies
- Programming language: Python
- GUI framework: Tkinter
- Data processing: Pandas
- Machine learning library: Scikit-learn
- Feature extraction: CountVectorizer
- Classification algorithm: Multinomial Naive Bayes
- Dataset: UCI SMS Spam Collection

### 5. System Requirements

Hardware:
- Computer or laptop
- Minimum 4 GB RAM recommended

Software:
- Windows, Linux, or macOS
- Python 3
- Visual Studio Code or another Python editor

### 6. Methodology

Step 1: Dataset Collection
The UCI SMS Spam Collection dataset is used to obtain labelled messages.

Step 2: Data Preparation
The dataset is loaded, missing values and duplicate messages are removed, and invalid labels are filtered out.

Step 3: Data Splitting
The dataset is divided into training data and testing data using an 80/20 split.

Step 4: Feature Extraction
CountVectorizer converts text messages into numerical word-count features.

Step 5: Model Training
The Multinomial Naive Bayes algorithm learns patterns from the training data.

Step 6: Model Evaluation
The trained model predicts the labels of the test messages. Accuracy is calculated by comparing predictions with their actual labels.

Step 7: GUI Development
Tkinter provides a window where users can enter a message, check its classification, and clear the input.

### 7. System Architecture

User enters a message
        |
        v
Text preprocessing and feature extraction
        |
        v
Trained Naive Bayes model
        |
        v
Spam or Not Spam prediction
        |
        v
Result displayed in GUI

### 8. Features
- User-friendly graphical interface
- Text message input
- Spam and not-spam predictions
- Test accuracy display
- Clear input functionality
- Machine learning-based classification

### 9. Results and Testing
During the current test run, the model achieved 98.16% accuracy on the held-out test dataset.

The application can be tested using normal messages and spam-like messages. The GUI displays the predicted category.

The reported accuracy applies to the selected dataset and test split. It does not guarantee that every new message will be classified correctly.

### 10. Limitations
- The current dataset contains SMS messages, not full email messages.
- Some legitimate messages may be classified as spam, and some spam messages may be missed.
- The model relies on patterns learned from its training data.
- The application does not independently verify links, senders, or message authenticity.

### 11. Future Enhancements
- Train and evaluate the model using a genuine email dataset.
- Add precision, recall, and F1-score evaluation.
- Improve text preprocessing.
- Save the trained model to reduce startup time.
- Develop a more advanced web-based interface.
- Add email integration with appropriate security controls.

### 12. Conclusion
The Spam Message Classifier demonstrates how machine learning can be used for text classification. It combines CountVectorizer, Multinomial Naive Bayes, and Tkinter to provide a working application with an interactive interface. The project provides practical experience in Python programming, data preparation, model evaluation, and GUI development.

### 13. References
1. UCI Machine Learning Repository — SMS Spam Collection:
   https://archive.ics.uci.edu/dataset/228/sms+spam+collection

2. Scikit-learn documentation:
   https://scikit-learn.org/

3. Python Tkinter documentation:
   https://docs.python.org/3/library/tkinter.html
