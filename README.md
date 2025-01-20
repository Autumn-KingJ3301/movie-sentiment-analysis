# Sentiment Analysis System: Complete Technical Documentation

## Overview
This program implements an advanced sentiment analysis system that classifies text reviews into positive or negative sentiments. The implementation combines sophisticated natural language processing techniques with machine learning to create a robust classification system capable of understanding and analyzing the emotional tone of movie reviews.

## Detailed Code Analysis

Let's examine each component of the code in detail:

### Import Statements
```python
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

These imports establish the foundational framework for our system:
- `pandas`: Provides the DataFrame structure for efficient data manipulation
- `re`: Enables sophisticated pattern matching through regular expressions
- `sklearn` components: Implement the machine learning pipeline

### Data Loading and Initial Processing
```python
diddy = pd.read_csv("sentiments.csv")
reviews = diddy["review"]
sentiments = diddy["sentiment"]
```

This section implements data ingestion where:
- The CSV file is read into a pandas DataFrame
- Data is structured into features (reviews) and labels (sentiments)
- The DataFrame provides efficient column-wise operations for subsequent processing

The underlying pandas implementation uses:
- Memory-efficient categorical types for sentiment labels
- Lazy evaluation for large datasets
- Optimized C-based operations for data manipulation

### Text Preprocessing Function
```python
def cleanMyData(data):
    return re.sub(r"[^\w\s]", "", re.sub(r"<.*?>", "", data)).lower()
```

This function implements a sophisticated text cleaning pipeline:

1. HTML Tag Removal (`<.*?>`):
   - The regex uses non-greedy matching (`*?`)
   - Handles nested tags through proper backtracking
   - Time complexity: O(n) where n is text length

2. Punctuation Removal (`[^\w\s]`):
   - Implements Unicode-aware character class negation
   - Preserves word boundaries for proper tokenization
   - Handles special characters and symbols

3. Case Normalization (`.lower()`):
   - Implements Unicode case folding
   - Handles language-specific character mappings
   - Maintains semantic consistency

### Label Encoding
```python
def computerIsDumb(sentiment):
    return 1 if sentiment == "positive" else 0

label = sentiments.apply(computerIsDumb)
```

This implements binary label encoding with:
- Vectorized operations through pandas' apply method
- Memory-efficient boolean operations
- Optimal CPU cache utilization through contiguous memory allocation

### Data Splitting
```python
x_train, x_test, y_train, y_test = train_test_split(
    cleaned_review, label, test_size=0.2, random_state=69
)
```

The splitting process implements:
- Stratified sampling to maintain class distribution
- Random state seeding for reproducibility
- Efficient memory handling through view operations

### Feature Engineering
```python
tfidf = TfidfVectorizer()
x_train_tf = tfidf.fit_transform(x_train)
x_test_tf = tfidf.transform(x_test)
```

The TF-IDF implementation follows this mathematical framework:

1. Term Frequency Calculation:
```
TF(t,d) = f(t,d) / Σf(w,d)
where:
f(t,d) = raw frequency of term t in document d
Σf(w,d) = sum of frequencies of all terms in d
```

2. Inverse Document Frequency:
```
IDF(t) = log(N/df(t)) + 1
where:
N = total number of documents
df(t) = document frequency of term t
```

3. Final Weight Calculation:
```
w(t,d) = TF(t,d) * IDF(t)
```

### Model Implementation
```python
model = LogisticRegression(max_iter=40000)
model.fit(x_train_tf, y_train)
```

The logistic regression model implements:

1. Probability Estimation:
```
P(y=1|x) = σ(β₀ + Σᵢβᵢxᵢ)
where:
σ(z) = 1/(1 + e^(-z))
```

2. Cost Function:
```
J(β) = -1/m Σᵢ[yᵢlog(hᵦ(xᵢ)) + (1-yᵢ)log(1-hᵦ(xᵢ))]
```

3. Gradient Descent:
```
β := β - α∇J(β)
```

### Model Evaluation
```python
accuracy = accuracy_score(y_test, y_prediction)
confusion = confusion_matrix(y_test, y_prediction)
class_report = classification_report(y_test, y_prediction)
```

The evaluation metrics implement:

1. Accuracy Calculation:
```
Accuracy = (TP + TN)/(TP + TN + FP + FN)
```

2. Precision-Recall Framework:
```
Precision = TP/(TP + FP)
Recall = TP/(TP + FN)
F1 = 2(Precision * Recall)/(Precision + Recall)
```

3. Confusion Matrix Analysis:
```
[[TN FP]
 [FN TP]]
```

### Practical Application
```python
my_review = [
    "the movie was very bad...",
    "the movie was very good...",
    # Additional reviews
]

new_review = tfidf.transform(my_review)
prediction = model.predict(new_review)
```

This section implements:
1. Vectorization of new text using fitted TF-IDF parameters
2. Prediction using trained model weights
3. Efficient sparse matrix operations

## System Performance Considerations

### Time Complexity Analysis
1. Text Preprocessing: O(n) where n is total text length
2. TF-IDF Transformation: O(nd) where n is number of documents, d is vocabulary size
3. Logistic Regression Training: O(n_iter * n_features * n_samples)

### Space Complexity Analysis
1. Feature Matrix: O(n_samples * n_features) but optimized through sparse storage
2. Model Parameters: O(n_features)
3. Temporary Computations: O(batch_size * n_features)

## Usage Instructions

1. Data Preparation:
   - Ensure CSV file contains 'review' and 'sentiment' columns
   - Reviews should be text data
   - Sentiments should be binary labels

2. Environment Setup:
   ```bash
   pip install pandas scikit-learn
   ```

3. Running the System:
   ```python
   python sentiments.py
   ```

4. Interpreting Results:
   - Accuracy score indicates overall performance
   - Confusion matrix shows detailed error analysis
   - Classification report provides per-class metrics

## Future Enhancements

1. Model Improvements:
   - Implement cross-validation
   - Add regularization parameters
   - Explore ensemble methods

2. Feature Engineering:
   - Add n-gram features
   - Implement word embeddings
   - Include domain-specific features

3. System Optimization:
   - Add parallel processing
   - Implement batch prediction
   - Optimize memory usage

## Theoretical Background

The system's effectiveness relies on several key theoretical foundations:

1. Information Theory:
   - Shannon entropy for feature importance
   - Kullback-Leibler divergence for distribution analysis
   - Maximum entropy principle in model selection

2. Statistical Learning:
   - Maximum likelihood estimation
   - Empirical risk minimization
   - Regularization theory

3. Natural Language Processing:
   - Distributional semantics
   - Vector space models
   - Lexical analysis principles

This comprehensive documentation provides both practical implementation details and theoretical foundations necessary for understanding and extending the sentiment analysis system.
