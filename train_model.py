import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
df_fake = pd.read_csv('Fake.csv')
df_real = pd.read_csv('True.csv')

# Add labels
df_fake['label'] = 0
df_real['label'] = 1

# Combine and shuffle
df = pd.concat([df_fake, df_real], ignore_index=True).sample(frac=1, random_state=42)

# Features and target
X = df['text']
y = df['label']

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
