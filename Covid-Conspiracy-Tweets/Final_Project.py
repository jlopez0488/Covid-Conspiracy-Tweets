import streamlit as st
import pandas as pd

#import numpy as np
#import itertools
# import snscrape.modules.twitter as sntwitter
# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#from wordcloud import WordCloud, STOPWORDS
# import nltk
# import nltk.data
# import re
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
# import gensim
# from gensim.utils import simple_preprocess
# from gensim.parsing.preprocessing import STOPWORDS
# import keras
# from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
# from tensorflow.keras.models import Model

menu = ['Home', 'EDA', 'Model Prediction','Future Work']
choice = st.sidebar.selectbox('Lets try', menu)

# Create the Home page
if choice == 'Home':
    st.video("https://youtu.be/MwDCyNpe-cc")
    st.title("Wuhan Lab Leak: From Conspiracy Theory to Federal Investigation — But How?")
    st.header("Using data analysis to understand how a fringe idea became the subject of a federal inquiry and the focus of newsrooms across the world")
    st.markdown("Since the start of the pandemic, a persistent question has dogged almost all efforts to understand and respond to the pandemic: is COVID-19 a naturally occurring disease or is the disease a human invention? \n Voices at a range of levels and expertise have weighed in on the question—including famously Donald Trump himself who would openly used the term China virus in tweets on the subject.")
    st.write("Indeed, it is Twitter where the question has been taken more seriously than anywhere else. A cadre of self appointed, amateur medical investigators have taken it upon themselves investigate the origins of the pandemic, specifically the theory that COVID-19 and the existing narrative around it exists as cover to a more sinister truth: the Chinese were working on a bioweapon that escaped the lab and appears in the world today at the coronavirus.")
    st.image("https://image.taiwannews.com.tw/photos/2021/02/18/1613638175-602e2a1f2496f.jpg")
    st.write("The subject appears to have come full circle with the newest occupant of the White House. After a rising chorus of concerned medical professionals and organizations have expressed keen interest in applying serious investigation in the Wuhan Lab Leak theory and legitimate news institutions including the New York Times, the BBC and other have begun to cover the question with earnest, U.S. President Joe Biden has called for a formal government investigation in to the question seemingly formalizing this idea’s move from conspiracy theory to legitimate question worth institutional study.")
    st.write("This project does not attempt to answer this specific question, but rather to apply data analysis to develop these observations through an analysis of over 17,000 tweets take from January to March 20201 using the “Wuhan lab leak” search term:")
    st.markdown("- The Wuhan Lab Leak seems like a persistent feature of the COVID conversation, but in fact the Twittersphere has been very quiet on this idea until very, very recently.")
    st.markdown("- A topic model analysis of the tweets shows a far greater interest and volume of activity in terms associated with agreement on the theory rather than those that suggest scrutiny or criticism of it.")
    st.markdown("- The Twitter search term “Wuhan lab leak” has users with the most activity in the U.S., the U.K. and adjacent areas, and India.")
    
    #second page
elif choice == "EDA":
    st.header("Data Collection & EDA")
    st.title("Getting and cleaning the Tweets")
    
    st.write("Let's get a look at the Tweets.")

    # if run:
    #     df = pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper('"Wuhan Lab Leak"').get_items(), 10))

    #     st.write(df)

    # st.write("What columns are in there?")

    # run1 = st.button("df.columns")
    # if run1:
        
    #     st.write(pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper('"Wuhan Lab Leak"').get_items(), 10)).columns)

    # st.write("For me, the most important and illustrative piece of what's here are the times, the tweet content and the location.")

    # run2 = st.button("Get the right columns")

    # if run2:

    #     st.write(pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper('"Wuhan Lab Leak"').get_items(), 10))[['content','date']]) 
    
    
    st.markdown("---")
    ##EDA
    st.write("Below is a sample of my entire dataset, which contains 17,180 tweets. Let's look at the just the content of the Tweets to do some textual analysis. First, let's look at the content.")
    wu = pd.read_json("https://raw.githubusercontent.com/jlopez0488/Covid-Conspiracy-Tweets/master/data/BIG-text-query-tweets.json", lines=True)
    words=[]
    def getwords():
        i = 0
        for i in range(len(wu)):
            x = wu.iloc[i]['content']
            words.append(x)
            i = i + 1
        return words
    words = getwords()
    words = pd.DataFrame(words)
    words = words.rename(columns={0:"original"})
    st.write(words.head(5), use_column_width='always')
    
    #####
    
    st.write("Let's see what's in this column as a wordcloud.")
    wc = st.button("Show me a WordCloud")
    if wc:
        st.image("https://raw.githubusercontent.com/jlopez0488/Covid-Conspiracy-Tweets/master/media/wordcloud.png")
        
    st.write("Tweets are generally pretty short, but let's get a closer look at where the lengths of these tweets are centered around.")
    hist = st.button("show me a hist!")
    if hist:
        st.image("https://raw.githubusercontent.com/jlopez0488/Covid-Conspiracy-Tweets/master/media/hist.png")
        
    st.write("Of the data set, a number of users have made their location publically visible. The location of the tweets has been mapped out below.")

    data = pd.read_csv("https://raw.githubusercontent.com/jlopez0488/Covid-Conspiracy-Tweets/master/data/example_file.csv")
    df = pd.DataFrame(data)

    st.markdown("**The number of locations is:**")
    st.write(len(df))
    map = st.button("Show me a map")
    if map:
        st.map(df)

elif choice == "Model Prediction":
    st.title("Model architecture")
    st.write("My idea is thus: let's build a fake news detection model to see if any trends emerge. Do the tweets change in factfulness over time? Is there a cluster of true seeming tweets at any point?")
    st.write("We're going to use a bidirectional LSTM model to try to assign a factfulness value to these tweets. Let's look at the model architecture first.")
    st.image("/Users/JesusLopez-Gomez/Desktop/streamlit_demo/media/Model.png")
    st.write("Let's take a glance at the training set that is used to train this dataset.")
    st.image("/Users/JesusLopez-Gomez/Desktop/streamlit_demo/media/Real and Fake news.png")
    st.markdown("---")
    st.write("You can see from the chart below that factfulness is pretty evenly distributed except for in May where unsure and above Tweets are concentrated.")
    
    st.image("/Users/JesusLopez-Gomez/Desktop/streamlit_demo/media/Tweets over Time.png")
    st.image("https://raw.githubusercontent.com/jlopez0488/Covid-Conspiracy-Tweets/master/media/Factfulness%20over%20time.png")

    st.write("It would be hard to truly evaluate the effectiveness of this model without double checking each one of the factfulness judgements made by the model, but a quick look at a sample of the value outputs is promising in my view.")
    st.image("/Users/JesusLopez-Gomez/Desktop/streamlit_demo/media/True tweets.png")
    st.image("/Users/JesusLopez-Gomez/Desktop/streamlit_demo/media/False tweets.png")
    st.markdown("---")
    st.write("To get a better sense of how this data set changes in May, I pulled just that month's data from the dataset. What you'll see here is that May is way overrepresented in the dataset, especially the later part of the month.")
    st.image("/Users/JesusLopez-Gomez/Desktop/streamlit_demo/media/May hist.png")
    st.write("Activity on this hashtag is concentrated far at the back. As far as our model can understand, the factfulness of this subject occurs somewhere around the final week of May 2021.")
    st.markdown("---")
    st.write("To complement my understanding of the data, I decided to perform a Latent Dirchlect Analysis to the tweets.")
    st.write("Briefly, this is a kind of textual analysis that tries to understand how a series of texts are related but calculating word distributions and assessing a likelihood that a given word would appear in a document. The calculation the model makes are used to generate topics that emerge from a broad analysis of the documents, so this type of analysis is called Topic Modelling.")
    st.header("LDA topics across the whole dataset")
    st.write("Topic analysis across the top three largest subjects shows a huge amount of interest in having this theory confirmed or advancing it. Key words here are: scientist, virus, bat, conspiracy.")
    st.image("/Users/JesusLopez-Gomez/Desktop/streamlit_demo/media/LDAa1.png")
    st.image("/Users/JesusLopez-Gomez/Desktop/streamlit_demo/media/LDAa2.png")
    st.image("/Users/JesusLopez-Gomez/Desktop/streamlit_demo/media/LDAa3.png")
    st.write("The 10th largest topic shows words that seem to indicate skepticism toward the idea. Key words here are: circustantial, evidence and checker.")
    st.image("/Users/JesusLopez-Gomez/Desktop/streamlit_demo/media/LDAa4.png")
elif choice == "Future Work":
    st.header("How Can This Be Improved?")
    st.markdown("-Further analysis of the late March activity.")
    st.markdown("-Stronger implementation of LDA.")
    st.markdown("-More research in to how machine learning and data analysis have supported fake news-related inquiries like these.")
