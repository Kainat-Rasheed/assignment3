import streamlit as st
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit CSS for enhanced styling
st.markdown("""
    <style>
    body {
        background-image: url('path_to_your_image.jpg');  /* Background image */
        background-size: cover;  /* Cover the entire background */
        font-family: 'Arial', sans-serif;  /* Set a modern font */
        color: white;  /* Default text color */
    }
    .title {
        color: #003366;  /* Dark blue for title */
        font-size: 3rem;  /* Increased size of the title */
        text-align: center;  /* Center the title */
        font-weight: bold;  /* Bold title */
        margin-bottom: 20px;  /* Space below the title */
    }
    .stHeader {
        color: #003366;  /* Dark blue for headers */
        font-weight: bold;  /* Bold headers */
        font-size: 1.8rem;  /* Increased size for headers */
        margin-top: 20px;  /* Space above headers */
    }
    .stSubheader {
        color: #66bb6a;  /* Light green for subheaders */
        font-weight: bold;  /* Bold subheaders */
        font-size: 1.5rem;  /* Increased size for subheaders */
    }
    .stTextInput, .stSelectbox, .stSlider {
        background-color: rgba(173, 216, 230, 0.9);  /* Light blue background for input fields */
        border: 2px solid #87cefa;  /* Light blue border */
        border-radius: 5px;  /* Rounded corners */
        padding: 10px;  /* Padding inside input fields */
        color: #003366;  /* Dark blue text color */
        font-size: 1.2rem;  /* Increased size for input fields */
        font-weight: bold;  /* Bold text for input fields */
    }
    .stSlider {
        color: #003366;  /* Dark blue for slider text */
    }
    .stButton {
        background-color: transparent;  /* No background color */
        color: #388e3c;  /* Text color */
        font-weight: bold;  /* Bold button text */
        border-radius: 5px;  /* Rounded corners */
        padding: 10px 20px;  /* Padding inside buttons */
        transition: background-color 0.3s;  /* Smooth transition for hover effect */
        font-size: 1.2rem;  /* Increased size for buttons */
    }
    .stButton:hover {
        background-color: #c8e6c9;  /* Light green on hover */
    }
    .stSidebar {
        background-color: #e0f7fa;  /* Light cyan background for the sidebar */
    }
    .sidebar .sidebar-content {
        background-color: #e9f5e9;  /* Light green background for sidebar */
    }
    .sidebar .stRadio {
        color: #003366;  /* Dark Blue for radio button labels */
    }
    .sidebar .stRadio > div > label {
        color: #20b2aa;  /* Light Sea Green for the radio button text */
        font-weight: bold;  /* Bold radio button text */
        font-size: 1.2rem;  /* Increased size for radio button text */
    }
    .sidebar .stRadio div {
        border: 2px solid #003366;  /* Dark Blue border around the radio buttons */
        border-radius: 5px;  /* Rounded corners */
        padding: 10px;  /* Padding inside radio buttons */
    }
    .stTextArea {
        background-color: rgba(173, 216, 230, 0.9);  /* Light blue background for text area */
        border: 2px solid #87cefa;  /* Light blue border */
        border-radius: 5px;  /* Rounded corners */
        padding: 10px;  /* Padding inside text area */
        color: #003366;  /* Dark blue text color */
        font-size: 1.2rem;  /* Increased size for text area */
        font-weight: bold;  /* Bold text for text area */
    }
    </style>
""", unsafe_allow_html=True)

# Task 1: 
st.markdown("<h1 class='title' style='color: darkblue; font-weight: bold;'>Spam Email Detection Dashboard</h1>", unsafe_allow_html=True)  # Dark Blue
st.markdown("<h2 style='color: #20b2aa; font-weight: bold;'>Welcome to the Interactive Email Spam Detector</h2>", unsafe_allow_html=True)  # Sea Green

st.subheader("What is Streamlit?")
st.markdown("Streamlit is a powerful Python framework that allows easy creation of interactive web applications for data science and machine learning.")

# Task 2:
# Text input for name with a styled label
st.markdown("<b style='color: #20b2aa;'>Enter your name:</b>", unsafe_allow_html=True)
name = st.text_input("", key="name_input")  # Leave the label empty for the input field

# Submit button
if st.button("Submit", key="submit_button"):
    # Displaying the output in Dark Blue
    st.markdown(f"<p style='color: #003366; font-weight: bold;'>Hello, {name}! Welcome to the spam detection app.</p>", unsafe_allow_html=True)

programming_languages = ["Python", "Java", "JavaScript", "C++", "Ruby"]

# Styled label for the selectbox
st.markdown("<b style='color: #20b2aa;'>Choose your favorite programming language:</b>", unsafe_allow_html=True)

# Selectbox for programming languages
favorite_language = st.selectbox("", programming_languages)  # Leave the label empty

# Displaying the output in Dark Blue
st.markdown(f"<p style='color: #003366; font-weight: bold;'>Your favorite programming language is: {favorite_language}</p>", unsafe_allow_html=True)

# Task 3: 
# Styled label for the slider
st.markdown("<b style='color: #20b2aa;'>Select a number (1-100):</b>", unsafe_allow_html=True)
number = st.slider("", 1, 100)  # Leave the label empty

# Displaying the output in Dark Blue
st.markdown(f"<p style='color: #003366; font-weight: bold;'>You selected: {number}</p>", unsafe_allow_html=True)

# Checkbox for displaying a message
if st.checkbox("Check this box to display a message"):
    st.markdown("<p style='color: #003366; font-weight: bold;'>Thank you for interacting!</p>", unsafe_allow_html=True)

# Radio button for skill level
st.markdown("<b style='color: #20b2aa;'>Select your skill level:</b>", unsafe_allow_html=True)
skill_level = st.radio("", ["Beginner", "Intermediate", "Advanced"])  # Leave the label empty

# Displaying the selected skill level in Dark Blue
st.markdown(f"<p style='color: #003366; font-weight: bold;'>You selected: {skill_level}</p>", unsafe_allow_html=True)

# Task 4: 

# Preprocess text (including stemming)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    stop_words = set(stopwords.words('english'))  # Stopwords
    tokens = word_tokenize(text)  # Tokenization
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]  # Stemming and removing stopwords
    return ' '.join(tokens)

# Load default dataset from email.csv
@st.cache_data
def load_default_data():
    try:
        df = pd.read_csv("email.csv")  # Read from email.csv
        if 'Message' in df.columns and 'Category' in df.columns:
            df['cleaned_text'] = df['Message'].apply(preprocess_text)
            df['Category'] = df['Category'].replace('ham', 'Not Spam')  # Replace ham with Not Spam
            return df
        else:
            return None
    except:
        return None

# File uploader with styled label
st.markdown("<h3 style='color: #20b2aa; font-weight: bold;'>Upload a CSV file of emails:</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"])  # Leave the label empty

if uploaded_file:
    df = pd.read_csv(uploaded_file)  # Read User CSV
    df['Category'] = df['Category'].replace('ham', 'Not Spam')  # Replace ham with Not Spam
else:
    df = load_default_data()  # Use Default email.csv if no file is uploaded

if df is not None:
    # Styled subheader for dataset preview
    st.markdown("<h2 style='color: #003366; font-weight: bold;'>Cleaned data</h2>", unsafe_allow_html=True)  # Dark Blue
    st.dataframe(df.head())

    # Bar chart for spam vs not spam distribution
    spam_count = df['Category'].value_counts()
    fig = px.bar(spam_count, x=spam_count.index, y=spam_count.values, 
                 title="<span style='color: #003366; font-weight: bold;'>Spam vs Not Spam Emails</span>", 
                 labels={'x': 'Category', 'y': 'Count'}, 
                 template="plotly_white")  # Use a white background for better visibility
    fig.update_traces(marker_color='rgba(0, 51, 102, 0.8)')  # Dark color for bars
    st.plotly_chart(fig)

    # WordCloud for spam emails
    spam_emails = ' '.join(df[df['Category'] == 'spam']['cleaned_text'])
    if spam_emails:  # Check if spam_emails is not empty
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_emails)

            # Check if wordcloud has been generated properly
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.markdown("<h3 style='color: #003366; font-weight: bold;'>WordCloud for Spam Emails</h3>", unsafe_allow_html=True)
            st.pyplot(plt)
            plt.close()
        except Exception as e:
            st.error(f"Error generating word cloud: {e}")
    else:
        st.warning("No spam emails available to generate a WordCloud.")
    
    # Interactive category filtering
    st.markdown("<h4 style='color: #003366; font-weight: bold;'>Select Category</h4>", unsafe_allow_html=True)
    category_filter = st.selectbox("Select Category", df['Category'].unique(), key="category_filter_visualization")  # Updated key for uniqueness
    filtered_df = df[df['Category'] == category_filter]
    st.dataframe(filtered_df.style.set_table_attributes('style="color: #003366; font-weight: bold;"'))  # Set color and bold for dataframe text

# Task 6: 
st.sidebar.markdown("<h3 style='color: #003366; font-weight: bold; font-size: 1.5rem;'>Spam Detection System</h3>", unsafe_allow_html=True)  # Dark Blue
page = st.sidebar.radio("Go to", ["Spam Detector", "Visualization"])

# If the selected page is "Spam Detector"
if page == "Spam Detector":
    if df is not None:
        st.subheader("Train and Test Spam Detector")

        # Train Naïve Bayes Model
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['cleaned_text'])
        y = df['Category']
        model = MultinomialNB()
        model.fit(X, y)

        # Email input for spam classification
        user_email = st.text_area("Enter an email to check if it's spam:", key="email_input")  # Added key for uniqueness

        if st.button("Classify Email"):
            # Preprocess the input email
            processed_email = preprocess_text(user_email)  # Preprocess the email text
            email_vector = vectorizer.transform([processed_email])  # Transform the preprocessed email
            
            # Make a prediction
            prediction = model.predict(email_vector)  # Get prediction
            prediction_label = prediction[0]  # Extract the prediction label
            
            # Display the prediction result
            st.markdown(f"<p style='color: #003366; font-weight: bold;'>The email is classified as: <span style='color: #ff0000;'>{prediction_label}</span></p>", unsafe_allow_html=True)
