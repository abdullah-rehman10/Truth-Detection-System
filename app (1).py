import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load model and vectorizer
model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load and prepare data
df_fake = pd.read_csv('fake.csv')
df_real = pd.read_csv('True.csv')
df_fake['label'] = 0
df_real['label'] = 1
df = pd.concat([df_fake, df_real], ignore_index=True).sample(frac=1, random_state=42)

X = df['text']
y = df['label']
X_vec = vectorizer.transform(X)
y_pred = model.predict(X_vec)

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.selectbox("Go to", ["Detector", "Statistics", "About Project", "About Us"])

# 1️⃣ Fake News Detector Page
if page == "Detector":
    st.title("📰 Fake News Detector")
    st.markdown("""
    Welcome to the **Fake News Detector**! 🧠  
    Use this app to test whether a news article is likely to be **Fake** 🟥 or **Real** 🟩.
    
    Paste your article below, or click **Surprise Me** to load a random news sample from our dataset.
    """)

    text_input = st.text_area("📄 Paste the news article content here:")

    if st.button("🚀 Detect"):
        if text_input.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            vec_input = vectorizer.transform([text_input])
            prediction = model.predict(vec_input)[0]
            prob = model.predict_proba(vec_input).max()

            label = "🟥 FAKE News" if prediction == 0 else "🟩 REAL News"
            st.success(f"### Prediction: {label}")
            st.markdown(f"**Confidence Level:** `{prob * 100:.2f}%`")

    st.markdown("---")
    st.write("### 🎲 Or try a surprise article!")

    if st.button("Surprise Me with a Random Article"):
        random_row = df.sample(1).iloc[0]
        sample_text = random_row['text']
        actual_label = "🟥 FAKE" if random_row['label'] == 0 else "🟩 REAL"

        st.text_area("Random Article:", sample_text, height=300)

        vec_input = vectorizer.transform([sample_text])
        prediction = model.predict(vec_input)[0]
        prob = model.predict_proba(vec_input).max()
        pred_label = "🟥 FAKE News" if prediction == 0 else "🟩 REAL News"

        st.success(f"### Model Prediction: {pred_label}")
        st.info(f"📌 **Actual Label (from dataset):** {actual_label}")
        st.markdown(f"**Confidence Level:** `{prob * 100:.2f}%`")


# 2️⃣ Statistics Page
elif page == "Statistics":
    st.title("📊 Model Performance Statistics")

    # Accuracy
    acc = accuracy_score(y, y_pred)
    st.write(f"### ✅ Accuracy: **{acc * 100:.2f}%**")

    # Confusion Matrix
    st.write("#### 📉 Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig_cm)

    # Classification Report Table
    st.write("#### 📋 Classification Report:")
    report = classification_report(y, y_pred, target_names=["Fake", "Real"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0))

    # Bar Plot of Precision, Recall, F1-score
    st.write("#### 📊 Visualizing Precision, Recall, F1-score")
    metrics_df = report_df.iloc[:2][["precision", "recall", "f1-score"]]
    fig_metrics, ax_metrics = plt.subplots()
    metrics_df.plot(kind='bar', ax=ax_metrics)
    plt.title("Precision, Recall, F1-score by Class")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    st.pyplot(fig_metrics)

    # Pie Chart of Predictions
    st.write("#### 🧁 Prediction Distribution:")
    pred_counts = pd.Series(y_pred).value_counts()
    pred_labels = ['Fake (0)', 'Real (1)']
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(pred_counts, labels=pred_labels, autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#99FF99'])
    ax_pie.axis('equal')
    st.pyplot(fig_pie)


# 3️⃣ About the Project Page
elif page == "About Project":
    st.title("📘 About the Project")
    st.markdown("""
    ### 🧠 What is Fake News Detection?
    Fake news refers to false or misleading information presented as news. It spreads rapidly and has the power to influence public opinion, elections, and global events.

    This project uses **Natural Language Processing (NLP)** and **Machine Learning (ML)** to detect such news.

    ### 🔨 How Was This Built?
    - **Data Source:** `True.csv` and `fake.csv` datasets from Kaggle
    - **Text Vectorization:** TfidfVectorizer
    - **Model:** Multinomial Naive Bayes
    - **Evaluation:** Accuracy, Confusion Matrix, Classification Report
    - **Frontend:** Streamlit

    ### ❓ Frequently Asked Questions

    **Q1: What does the model analyze?**  
    ➤ It processes the article text and extracts word patterns that correlate with real or fake articles.

    **Q2: Is this 100% accurate?**  
    ➤ No ML model is perfect. This one is trained on a limited dataset. Results are suggestive, not conclusive.

    **Q3: Can this be used on social media posts?**  
    ➤ Technically yes, but short text may reduce accuracy. You can try pasting tweets or captions.

    **Q4: Can the model learn new trends?**  
    ➤ Not automatically. It needs retraining with updated datasets.

    ### 📈 Applications
    - Journalism verification
    - Educational tools
    - Browser extensions (future idea)
    - Academic research
    """)

# 4️⃣ About Us Page
elif page == "About Us":
    st.title("👥 About Us")
    st.markdown("""
    This project was developed by a passionate team of students combining skills in machine learning, data analysis, and web development.

    ### 👨‍💻 Team Members
    - **Ibrahim Naveed**
    - **Rana Mohsin Ali**
    - **Mian Hashim Naveed**

    ### 🛠 What We Used
    - **Python**, **pandas**, **scikit-learn**
    - **Naive Bayes Classification**
    - **TfidfVectorizer** for feature extraction
    - **Streamlit** for the user interface

    Our goal was to build an intuitive tool that highlights how machine learning can be applied to real-world problems like fake news detection.
    """)
