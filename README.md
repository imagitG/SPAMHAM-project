Project on classifying emails as SPAM or HAM

steps:
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
