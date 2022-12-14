# Deploying the Sklearn NMF model & Gensim LDA model using Streamlit
# Importing libraries
import sklearn

import pickle

import streamlit as st
from streamlit import components

# Loading the NMF model & TF-IDF vectorizer
nmf_path = "nmf/sk_nmf_model.pkl"
sk_nmf = pickle.load(open(nmf_path, "rb"))

tf_path = "nmf/tfidf_vectorizer.pkl"
tfidf_vectorizer = pickle.load(open(tf_path, "rb"))

# ANd the document-term matrix
dtm_path = "nmf/dtm.pkl"
dtm_tfidf = pickle.load(open(dtm_path, "rb"))

# Display each model's visualizations.
st.title("Topic Modelling Case Study")
st.markdown("I trained 3 Topic Modelling algorithms; LDA, NMF and BERTopic, to describe the topics present in the \
    BBC articles dataset. This dataset is a collection of news articles in 5 areas; sport, politics, technology,\
    business and entertainment. Below, we see how each model has differentiated the topics and distributed the \
    keywords within them.")

st.markdown("P.S. Choose the Light Theme in Settings to view the Topic-Word Distributions more clearly.")

st.subheader("Non-Negative Matrix Factorization Model (Sklearn)")

# NMF model
import pyLDAvis.sklearn

prepared_pyLDAvis_data = pyLDAvis.sklearn.prepare(sk_nmf, dtm_tfidf, tfidf_vectorizer)


# Exporting the visualization as an iframe 
html_string = pyLDAvis.prepared_data_to_html(prepared_pyLDAvis_data)
# print(html_string)
components.v1.html(html_string, width = 1300, height = 1000, scrolling = True)

# LDA Model
from gensim.models import LdaMulticore

path = "lda/"
lda_model = LdaMulticore.load(path + "/lda_model")

# LDA's tf-idf corpus & dictionary
from gensim.corpora.dictionary import Dictionary

id2word = Dictionary.load(path + "/lda_model.id2word")

p = "lda/tfidf_corpus_gn.pkl"

tfidf_corpus = pickle.load(open(p, "rb"))

# Visualizing lda model's topic distribution
import pyLDAvis.gensim_models as gensimvis

lda_vis = gensimvis.prepare(lda_model, tfidf_corpus, id2word)


# Exporting the pyLDAvis visualization as an HTML iframe
st.subheader("Latent Dirichlet Allocation Model (Gensim)")

html_lda = pyLDAvis.prepared_data_to_html(lda_vis)
components.v1.html(html_lda, width = 1300, height = 1000, scrolling = True)
