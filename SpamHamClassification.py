import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support as score
import string

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Define stopwords and Porter Stemmer
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

# Load dataset
dataset = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None)
dataset.columns = ['label', 'body_text']

# Encode labels: 'ham' -> 0, 'spam' -> 1
le = LabelEncoder()
dataset['label'] = le.fit_transform(dataset['label'])

# Function to calculate punctuation percentage
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / (len(text) - text.count(" ")), 3) * 100

# Add body length and punctuation percentage columns
dataset['body_len'] = dataset["body_text"].apply(lambda x: len(x) - x.count(" "))
dataset['punct%'] = dataset['body_text'].apply(lambda x: count_punct(x))

# Function to clean text: remove punctuation, tokenize, remove stopwords, apply stemming
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)  # Tokenize based on non-word characters
    text = [ps.stem(word) for word in tokens if word not in stopwords and word != '']
    return text

# Apply TF-IDF Vectorizer with custom clean_text function
tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(dataset['body_text'])

# Concatenate TF-IDF features with body_len and punct%
X_features = pd.concat([dataset['body_len'], dataset['punct%'], pd.DataFrame(X_tfidf.toarray())], axis=1)

# Convert all feature names (columns) to strings to avoid errors
X_features.columns = X_features.columns.astype(str)

# Model using K-Fold cross-validation
rf = RandomForestClassifier(n_jobs=-1)  # Allow parallelism
k_fold = KFold(n_splits=5)

accuracy = cross_val_score(rf, X_features, dataset['label'], cv=k_fold, scoring="accuracy", n_jobs=-1)

# New Section: Train-Test Split and Model Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_features, dataset['label'], test_size=0.3, random_state=0)

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=500, max_depth=20, n_jobs=-1)
rf_model = rf.fit(X_train, y_train)

# Print the top 10 most important features with better formatting
print("\nTop 10 Important Features:")
for importance, feature in sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[:10]:
    print(f"Feature: {feature:>10} | Importance: {importance:.4f}")

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate precision, recall, fscore, and support for the model
precision, recall, fscore, support = score(y_test, y_pred, pos_label=1, average='binary')

# Print Precision, Recall, F-score, and Accuracy with better formatting
print("\nModel Evaluation Metrics:")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F-score:   {fscore:.3f}")
print(f"Accuracy:  {accuracy.mean():.3f}")


'''

-------------------------------------------------------------------------------------------------
OUTPUT: 
Top 10 Important Features:
Feature:   body_len | Importance: 0.0532
Feature:       1803 | Importance: 0.0391
Feature:       7352 | Importance: 0.0354
Feature:       4798 | Importance: 0.0316
Feature:       3134 | Importance: 0.0255
Feature:       2031 | Importance: 0.0236
Feature:       5726 | Importance: 0.0210
Feature:       7029 | Importance: 0.0186
Feature:       6287 | Importance: 0.0179
Feature:       6748 | Importance: 0.0174

Model Evaluation Metrics:
Precision: 1.000
Recall:    0.564
F-score:   0.721
Accuracy:  0.974

-------------------------------------------------------------------------------------------------
Explanation:

1   Setup and Imports:

    Import necessary libraries (nltk, pandas, re, sklearn).
    Set pandas to display all columns.
    Download NLTK stopwords.

2   Data Loading and Preprocessing:

    Load the dataset from a TSV file (SMSSpamCollection.tsv) with columns label and body_text.
    Encode labels: Convert 'ham' to 0 and 'spam' to 1 using LabelEncoder.

3   Feature Engineering:

    Punctuation Percentage: Calculate the percentage of punctuation characters in each SMS body.
    Body Length: Compute the length of the SMS body excluding spaces.
    Add these features (body_len and punct%) to the dataset.

4   Text Cleaning:

    Function clean_text: Removes punctuation, tokenizes the text, removes stopwords, and applies stemming using the Porter Stemmer.

5   TF-IDF Vectorization:

    Apply TfidfVectorizer with the custom clean_text function to transform SMS text into TF-IDF features.

6   Feature Concatenation:

    Combine TF-IDF features with body_len and punct% into a single DataFrame (X_features).

7   Model Training and Evaluation:

    K-Fold Cross-Validation:    Use Random Forest Classifier with 5-fold cross-validation to estimate model accuracy.
    Train-Test Split:           Split the data into training and testing sets (70% training, 30% testing).
    Train Model:                Fit a Random Forest Classifier with 500 estimators and a maximum depth of 20 on the training data.
    Feature Importance:         Print the top 10 most important features as determined by the Random Forest model.
    Prediction and Metrics:     Make predictions on the test data and calculate precision, recall, F-score, and accuracy.
--------------------------------------------------------------------------------------------------------------------------------------

'''