import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import string
import nltk
from nltk.corpus import stopwords

def preprocess_text(text):
    text = text.lower()
    
    text = "".join([ch for ch in text if ch not in string.punctuation])
    
    text = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE).sub(r'', text)
    
    text = " ".join([word for word in text.split() if word not in stopwords_list])
    
    return text

def predict_sentiment(text):
    text = preprocess_text(text)

    tfidf_feat = tfidf_vec.transform([text])

    embed_feat = labse_model.encode([text])

    combined_feat = np.append(embed_feat[0], tfidf_feat.toarray()[0]).reshape(1, -1)
    prediction = svm_model.predict(combined_feat)[0]

    return label_reverse.get(prediction, "Unknown")

if __name__ == "__main__":

    nltk.download('stopwords', quiet=True)
    stopwords_list = stopwords.words('english')
    stopwords_list.extend(['ena','unaku','per','irukanga','panna','yarum','mattum','ivan','ada','pesa','unakku','k','sari','idhu','vida','vittu','enga','yen','ithu','poda','dey','irundhu','ya','la', 'u','r','s','a','bro','da','dei','dai','nu','ah','nee','ni','illa','un','ok','na','pls','ur','unga'])
    stopwords_list.extend(['அந்த', 'இது', 'என்ன', 'என்', 'உங்கள்', 'உள்ளது', 'எனக்கு', 'இந்த', 'அவர்கள்', 'எனக்கு', 'இந்த', 'அவர்கள்', 'இந்த', 'அந்த', 'உங்கள்', 'என்', 'அனைத்து', 'இது', 'அவர்கள்', 'இந்த', 'அவர்கள்', 'இந்த', 'அவர்கள்', 'இந்த', 'அவர்கள்', 'இந்த', 'அவர்கள்', 'என்ன', 'என்', 'உங்கள்', 'என்ன', 'அவர்கள்', 'எனக்கு', 'என்', 'உங்கள்', 'இது', 'அந்த', 'இது', 'அவர்கள்', 'இது', 'அவர்கள்', 'இந்த', 'அவர்கள்', 'இந்த', 'அவர்கள்', 'இந்த', 'அவர்கள்', 'இந்த', 'அந்த', 'உங்கள்', 'என்', 'அனைத்து', 'இந்த', 'அவர்கள்', 'இந்த', 'அவர்கள்', 'இந்த', 'அவர்கள்', 'இந்த', 'அவர்கள்', 'இந்த', 'அவர்க'])
    stopwords_list.extend(['indha','antha','vera','iruka','iruku','pola','innum','avan','summa','ellam','thaan','romba','ana','ama','apdi','ithula','po','evlo','eruku','irukum','nama','enna','va','hi','h','ku','iruku','naa','va','oru','athu','avanga','neenga','tha','en','di','dhan','ne','ella','intha'])

    svm_model = joblib.load('linear_svm_model.joblib')
    tfidf_vec = joblib.load('tfidf_vectorizer.joblib')
    labse_model = SentenceTransformer("sentence-transformers/LaBSE")

    label_reverse = {
        0: 'Uknown State',
        1: 'Positive',
        2: 'Mixed Feelings',
        3: 'Negative'
    }

    st.title("🧠 Tamil-English Sentiment Classifier")
    st.write("Enter your sentence below and get the sentiment classification.")

    user_input = st.text_area("Your Input", placeholder="Type something in Tamil, English, or mixed...")

    if st.button("Analyze"):
        if user_input.strip():
            result = predict_sentiment(user_input)
            st.success(f"Predicted Sentiment: **{result}**")
        else:
            st.warning("Please enter some text to analyze.")