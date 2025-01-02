import pandas as pedo
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

diddy = pedo.read_csv("sentiments.csv")

reviews = diddy["review"]
sentiments = diddy["sentiment"]

tfidf = TfidfVectorizer()

def cleanMyData(data):
    return re.sub(r"[^\w\s]", "", re.sub(r"<.*?>", "", data)).lower()

def computerIsDumb(sentiment):
    return 1 if sentiment == "positive" else 0

cleaned_review = reviews.apply(cleanMyData)
label = sentiments.apply(computerIsDumb)

x_train, x_test, y_train, y_test = train_test_split(
    cleaned_review, label, test_size=0.2, random_state=69
)


x_train_tf = tfidf.fit_transform(x_train)
x_test_tf = tfidf.transform(x_test)

model = LogisticRegression(max_iter=40000)
model.fit(x_train_tf, y_train)

y_prediction = model.predict(x_test_tf)

accuracy = accuracy_score(y_test, y_prediction)
confusion = confusion_matrix(y_test, y_prediction)
class_report = classification_report(y_test, y_prediction)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(class_report)
print("\n ########## Actual Use ############")

my_review = [
    "the movie was very bad and it spoiled my whole mood",
    "the movie was very good and it made my whole mood happy, at this point I am just glazing over it",
    "I absolutely loved the film, it was fantastic!",
    "The plot was boring and predictable, I didn't enjoy it.",
    "An excellent movie with a great storyline and superb acting.",
    "Terrible movie, I wasted two hours of my life.",
    "It was an okay movie, not too bad but not great either.",
    "The cinematography was beautiful, but the story was lacking.",
    "A masterpiece, one of the best movies I've ever seen.",
    "I didn't like the movie at all, it was a huge disappointment."
]

new_review = tfidf.transform(my_review)
prediction = model.predict(new_review)
print(prediction)

for i in range(len(my_review)):
    print(f"Review: {my_review[i]} \n Sentiment: {'positive' if prediction[i] == 1 else 'negative'}")

