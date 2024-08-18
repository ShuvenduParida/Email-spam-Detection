import pickle
import numpy as np
import streamlit as st
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'spam_classifier_model.pkl')

# Assuming only the classifier was saved in the pickle file
with open(model_path, 'rb') as model_file:
    classifier = pickle.load(model_file)

# Manually load or define the word_dict if it's stored elsewhere or recreate it
with open('word_dict.pkl', 'rb') as word_dict_file:
    word_dict = pickle.load(word_dict_file)

# Streamlit UI
st.title("Email Spam Classifier")
st.write("Enter the content of the email below:")

# Text area for input
email_content = st.text_area("Email Content", height=200)

# Button to classify email
if st.button("Classify Email"):
    sample = []
    for i in word_dict:
        sample.append(email_content.split(" ").count(i[0]))

    sample = np.array(sample)
    prediction = classifier.predict(sample.reshape(1, 1000))

    if prediction == 1:
        st.error("This email is Spam!")
    else:
        st.success("This email is Ham (Not Spam).")
