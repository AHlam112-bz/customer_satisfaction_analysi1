import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from tensorflow.keras.models import load_model
import fasttext.util
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.utils import simple_preprocess
from PIL import Image
# Load the pre-trained FastText model
fasttext.util.download_model('en', if_exists='ignore')  
ft_model = fasttext.load_model('cc.en.300.bin')  # load the English model

# Load the pre-trained model
model = load_model('./model_1.h5')

patterns_to_remove = [r':\)', r':\(', r':D']
def remove_patterns(text):
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text)
    return text

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_htmltags(text):
    html_pattern = re.compile(r'<.*?>')  
    return html_pattern.sub('', text)

def remove_extra_spaces_and_numbers(text):
    text = ' '.join(text.split())
    text = re.sub(r'\d+', '', text)
    return text

def remove_punctuations(text):
    translates = str.maketrans("", "", string.punctuation)
    return text.translate(translates)

def remove_stopwords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

def lowercase(text):
    return text.lower()

def stem_text(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text

def clean_data(text):
    text = remove_patterns(text)
    text = remove_urls(text)
    text = remove_htmltags(text)
    text = remove_extra_spaces_and_numbers(text)
    text = remove_punctuations(text)
    text = remove_stopwords(text)
    text = lowercase(text)
    text = stem_text(text)
    return text

def predict_sentiment(comment):
    cleaned_comment = clean_data(comment)

    print("Cleaned Comment:", cleaned_comment)

    if not cleaned_comment:
        return "Unable to determine sentiment", 0, 0

    tokenized_comment = simple_preprocess(cleaned_comment)

    print("Tokenized Comment:", tokenized_comment)

    embeddings = []
    for token in tokenized_comment:
        vector = ft_model.get_word_vector(token)
        embeddings.append(vector)

    if embeddings:
        mean_comment_vector = np.mean(embeddings, axis=0)
    else:
        mean_comment_vector = np.zeros(ft_model.get_word_vector("example").shape)

    comment_data = np.array(mean_comment_vector)
    comment_data = comment_data.reshape((1, comment_data.shape[0], 1))

    print("Comment Data Shape:", comment_data.shape)

    
    prediction = model.predict(comment_data)[0][0]

    
    print("Raw Prediction:", prediction)

    positive_percentage = round(prediction * 100, 2)
    negative_percentage = round((1 - prediction) * 100, 2)

    sentiment = "Positive" if prediction >= 0.6 else "Negative"

    print("Sentiment:", sentiment)
    print("Positive Percentage:", positive_percentage)
    print("Negative Percentage:", negative_percentage)
    
    image_path = generate_pie_chart(positive_percentage, negative_percentage)

    return sentiment,image_path

def generate_pie_chart(positive_percentage, negative_percentage):
    labels = ['Positive', 'Negative']
    sizes = [positive_percentage, negative_percentage]
    colors = ['green', 'red']
    explode = (0.1, 0)  # explode 1st slice

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  
    
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img = Image.open(img_buf)

    plt.close()

    return img

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(),
    outputs=[
        gr.Textbox("text", label="Sentiment"),
        gr.Image(type="pil", label="Sentiment Pie Chart"),
    ],
    live=True,
    analytics_enabled=False,
    title="Sentiment Analysis",
    description="Enter a comment and get sentiment prediction. Positive and negative percentages are visualized.",
)

iface.launch(share=False)
