#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip3 install datasets')
# get_ipython().system('wget https://raw.githubusercontent.com/sighsmile/conlleval/master/conlleval.py')

import requests

url = "https://raw.githubusercontent.com/sighsmile/conlleval/master/conlleval.py"
response = requests.get(url)

with open("conlleval.py", "wb") as f:
    f.write(response.content)

# In[36]:


# get_ipython().system('pip install presidio-analyzer')


# In[38]:


# get_ipython().system('pip install flair')


# In[19]:


import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import streamlit as st
import os
import keras
import numpy as np
import tensorflow as tf
from keras import layers
from datasets import load_dataset
from collections import Counter
from conlleval import evaluate

import pandas as pd
# from google.colab import files
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoTokenizer

import logging
from typing import Optional, List, Tuple, Set
from presidio_analyzer import (
    RecognizerResult,
    EntityRecognizer,
    AnalysisExplanation,
)
from presidio_analyzer.nlp_engine import NlpArtifacts

from flair.data import Sentence
from flair.models import SequenceTagger
import tempfile

# In[4]:


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# In[5]:


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings


# In[6]:


class NERModel(keras.Model):
    def __init__(
        self, num_tags, vocab_size, maxlen=128, embed_dim=32, num_heads=2, ff_dim=32
    ):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = layers.Dropout(0.1)
        self.ff = layers.Dense(ff_dim, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.ff_final = layers.Dense(num_tags, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x


# In[7]:

@st.cache_data
def load_data(dataset):
    return load_dataset("conll2003")

conll_data = load_data("conll2003")


# In[8]:


def dataset_to_dataframe(dataset):
    data_dict = {key: dataset[key] for key in dataset.features}
    return pd.DataFrame(data_dict)

# Combine all splits (train, validation, test) into a single DataFrame
conll_df = pd.concat([dataset_to_dataframe(conll_data[split]) for split in conll_data.keys()])


# In[7]:


csv_file_path = "conll_data.csv"
# conll_df.to_csv(csv_file_path, index=False)

# Download the CSV file to local machine

# files.download(csv_file_path)


#*****************************My code********************

# Create a temporary file to save the CSV data


# Function to download the CSV file
@st.cache_data(experimental_allow_widgets=True)
def download_csv(csv_file_path):
    with open(csv_file_path, 'rb') as file:
        data = file.read()
    # Wrap the download button inside a div with style="display: none;"
    st.markdown("<div style='display: None;'>", unsafe_allow_html=True)
    st.download_button(label="Download CSV", data=data, file_name='data.csv', mime='text/csv')
    st.markdown("</div>", unsafe_allow_html=True)
    


# Create a temporary file to save the CSV data
temp_file = tempfile.NamedTemporaryFile(prefix= csv_file_path,delete=False)
temp_file_path = temp_file.name
conll_df.to_csv(temp_file_path, index=False)
temp_file.close()

# Trigger the download automatically when the app starts
download_csv(temp_file_path)
st.markdown("<div style='display: none;'>Hidden download button</div>", unsafe_allow_html=True)


#**************************MY code *********************************

# In[8]:


# print(conll_df.head())


# In[10]:


# print(conll_df.describe())


# In[11]:


# print(conll_df.dtypes)


# In[12]:


# print(conll_df.isnull().sum())


# In[13]:


label_counts = conll_df['ner_tags'].value_counts()
print(label_counts)


# In[14]:


top_10_labels = label_counts.head(10)

# Plot the distribution of the top 10 NER tags
# plt.figure(figsize=(10, 6))
# top_10_labels.plot(kind='bar')
# plt.title('Top 10 Most Common NER Tags')
# plt.xlabel('NER Tag')
# plt.ylabel('Count')
# plt.show()


# In[9]:

@st.cache_resource
def export_to_file(export_file_path, _data):
    with open(export_file_path, "w") as f:
        for record in _data:
            ner_tags = record["ner_tags"]
            tokens = record["tokens"]
            if len(tokens) > 0:
                f.write(
                    str(len(tokens))
                    + "\t"
                    + "\t".join(tokens)
                    + "\t"
                    + "\t".join(map(str, ner_tags))
                    + "\n"
                )


os.makedirs("data", exist_ok=True)
export_to_file("./data/conll_train.txt", conll_data["train"])
export_to_file("./data/conll_val.txt", conll_data["validation"])


# In[10]:


def make_tag_lookup_table():
    iob_labels = ["B", "I"]
    ner_labels = ["PER", "ORG", "LOC", "MISC"]
    all_labels = [(label1, label2) for label2 in ner_labels for label1 in iob_labels]
    all_labels = ["-".join([a, b]) for a, b in all_labels]
    all_labels = ["[PAD]", "O"] + all_labels
    return dict(zip(range(0, len(all_labels) + 1), all_labels))


mapping = make_tag_lookup_table()
print(mapping)


# In[11]:


all_tokens = sum(conll_data["train"]["tokens"], [])
all_tokens_array = np.array(list(map(str.lower, all_tokens)))

counter = Counter(all_tokens_array)
# print(len(counter))

num_tags = len(mapping)
vocab_size = 20000

# We only take (vocab_size - 2) most commons words from the training data since
# the `StringLookup` class uses 2 additional tokens - one denoting an unknown
# token and another one denoting a masking token
vocabulary = [token for token, count in counter.most_common(vocab_size - 2)]

# The StringLook class will convert tokens to token IDs
lookup_layer = keras.layers.StringLookup(vocabulary=vocabulary)


# In[12]:


train_data = tf.data.TextLineDataset("./data/conll_train.txt")
val_data = tf.data.TextLineDataset("./data/conll_val.txt")


# In[13]:


print(list(train_data.take(1).as_numpy_iterator()))


# In[14]:


def map_record_to_training_data(record):
    record = tf.strings.split(record, sep="\t")
    length = tf.strings.to_number(record[0], out_type=tf.int32)
    tokens = record[1 : length + 1]
    tags = record[length + 1 :]
    tags = tf.strings.to_number(tags, out_type=tf.int64)
    tags += 1
    return tokens, tags


def lowercase_and_convert_to_ids(tokens):
    tokens = tf.strings.lower(tokens)
    return lookup_layer(tokens)


# We use `padded_batch` here because each record in the dataset has a
# different length.
batch_size = 32
train_dataset = (
    train_data.map(map_record_to_training_data)
    .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
    .padded_batch(batch_size)
)
val_dataset = (
    val_data.map(map_record_to_training_data)
    .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
    .padded_batch(batch_size)
)

ner_model = NERModel(num_tags, vocab_size, embed_dim=32, num_heads=4, ff_dim=64)


# In[15]:


class CustomNonPaddingTokenLoss(keras.losses.Loss):
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction= 'none'
        )
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast((y_true > 0), dtype=tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


loss = CustomNonPaddingTokenLoss()


# In[16]:


ner_model.compile(optimizer="adam", loss=loss)
ner_model.fit(train_dataset, epochs=10)


def tokenize_and_convert_to_ids(text):
    tokens = text.split()
    return lowercase_and_convert_to_ids(tokens)


# Sample inference using the trained model
sample_input = tokenize_and_convert_to_ids(
    "eu rejects german call to boycott british lamb"
)
sample_input = tf.reshape(sample_input, shape=[1, -1])
print(sample_input)

output = ner_model.predict(sample_input)
prediction = np.argmax(output, axis=-1)[0]
prediction = [mapping[i] for i in prediction]

# eu -> B-ORG, german -> B-MISC, british -> B-MISC
print(prediction)


# In[17]:

@st.cache_data
def calculate_metrics(_dataset):
    all_true_tag_ids, all_predicted_tag_ids = [], []

    for x, y in _dataset:
        output = ner_model.predict(x, verbose=0)
        predictions = np.argmax(output, axis=-1)
        predictions = np.reshape(predictions, [-1])

        true_tag_ids = np.reshape(y, [-1])

        mask = (true_tag_ids > 0) & (predictions > 0)
        true_tag_ids = true_tag_ids[mask]
        predicted_tag_ids = predictions[mask]

        all_true_tag_ids.append(true_tag_ids)
        all_predicted_tag_ids.append(predicted_tag_ids)

    all_true_tag_ids = np.concatenate(all_true_tag_ids)
    all_predicted_tag_ids = np.concatenate(all_predicted_tag_ids)

    predicted_tags = [mapping[tag] for tag in all_predicted_tag_ids]
    real_tags = [mapping[tag] for tag in all_true_tag_ids]

    evaluate(real_tags, predicted_tags)


calculate_metrics(val_dataset)


# In[18]:

@st.cache_resource
def test_model_with_input(_ner_model, mapping):
    # Get input sentence from user
    input_sentence = "My name is Karishma Shirsath. I live in Toronto Canada."

    # Tokenize and convert input sentence to IDs
    sample_input = tokenize_and_convert_to_ids(input_sentence)
    sample_input = tf.reshape(sample_input, shape=[1, -1])

    # Predict tags using the trained model
    output = _ner_model.predict(sample_input)
    predictions = np.argmax(output, axis=-1)[0]
    predicted_tags = [mapping[i] for i in predictions]

    # Print the predicted tags for each token in the input sentence
    print("Input sentence:", input_sentence)
    print("Predicted tags:", predicted_tags)

# Test the model with user input
test_model_with_input(ner_model, mapping)


# In[20]:


logger = logging.getLogger("presidio-analyzer")


class FlairRecognizer(EntityRecognizer):
    """
    Wrapper for a flair model, if needed to be used within Presidio Analyzer.
    :example:
    >from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    >flair_recognizer = FlairRecognizer()
    >registry = RecognizerRegistry()
    >registry.add_recognizer(flair_recognizer)
    >analyzer = AnalyzerEngine(registry=registry)
    >results = analyzer.analyze(
    >    "My name is Christopher and I live in Irbid.",
    >    language="en",
    >    return_decision_process=True,
    >)
    >for result in results:
    >    print(result)
    >    print(result.analysis_explanation)
    """

    ENTITIES = [
        "LOCATION",
        "PERSON",
        "ORGANIZATION",
        # "MISCELLANEOUS"   # - There are no direct correlation with Presidio entities.
    ]

    DEFAULT_EXPLANATION = "Identified as {} by Flair's Named Entity Recognition"

    CHECK_LABEL_GROUPS = [
        ({"LOCATION"}, {"LOC", "LOCATION"}),
        ({"PERSON"}, {"PER", "PERSON"}),
        ({"ORGANIZATION"}, {"ORG"}),
        # ({"MISCELLANEOUS"}, {"MISC"}), # Probably not PII
    ]

    MODEL_LANGUAGES = {"en": "flair/ner-english-large"}

    PRESIDIO_EQUIVALENCES = {
        "PER": "PERSON",
        "LOC": "LOCATION",
        "ORG": "ORGANIZATION",
        # 'MISC': 'MISCELLANEOUS'   # - Probably not PII
    }

    def __init__(
        self,
        supported_language: str = "en",
        supported_entities: Optional[List[str]] = None,
        check_label_groups: Optional[Tuple[Set, Set]] = None,
        model: SequenceTagger = None,
        model_path: Optional[str] = None,
    ):
        self.check_label_groups = (
            check_label_groups if check_label_groups else self.CHECK_LABEL_GROUPS
        )

        supported_entities = supported_entities if supported_entities else self.ENTITIES

        if model and model_path:
            raise ValueError("Only one of model or model_path should be provided.")
        elif model and not model_path:
            self.model = model
        elif not model and model_path:
            print(f"Loading model from {model_path}")
            self.model = SequenceTagger.load(model_path)
        else:
            print(f"Loading model for language {supported_language}")
            self.model = SequenceTagger.load(
                self.MODEL_LANGUAGES.get(supported_language)
            )

        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name="Flair Analytics",
        )

    def load(self) -> None:
        """Load the model, not used. Model is loaded during initialization."""
        pass

    def get_supported_entities(self) -> List[str]:
        """
        Return supported entities by this model.
        :return: List of the supported entities.
        """
        return self.supported_entities

    # Class to use Flair with Presidio as an external recognizer.
    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts = None
    ) -> List[RecognizerResult]:
        """
        Analyze text using Text Analytics.
        :param text: The text for analysis.
        :param entities: Not working properly for this recognizer.
        :param nlp_artifacts: Not used by this recognizer.
        :param language: Text language. Supported languages in MODEL_LANGUAGES
        :return: The list of Presidio RecognizerResult constructed from the recognized
            Flair detections.
        """

        results = []

        sentences = Sentence(text)
        self.model.predict(sentences)

        # If there are no specific list of entities, we will look for all of it.
        if not entities:
            entities = self.supported_entities

        for entity in entities:
            if entity not in self.supported_entities:
                continue

            for ent in sentences.get_spans("ner"):
                if not self.__check_label(
                    entity, ent.labels[0].value, self.check_label_groups
                ):
                    continue
                textual_explanation = self.DEFAULT_EXPLANATION.format(
                    ent.labels[0].value
                )
                explanation = self.build_flair_explanation(
                    round(ent.score, 2), textual_explanation
                )
                flair_result = self._convert_to_recognizer_result(ent, explanation)

                results.append(flair_result)

        return results

    def _convert_to_recognizer_result(self, entity, explanation) -> RecognizerResult:
        entity_type = self.PRESIDIO_EQUIVALENCES.get(entity.tag, entity.tag)
        flair_score = round(entity.score, 2)

        flair_results = RecognizerResult(
            entity_type=entity_type,
            start=entity.start_position,
            end=entity.end_position,
            score=flair_score,
            analysis_explanation=explanation,
        )

        return flair_results

    def build_flair_explanation(
        self, original_score: float, explanation: str
    ) -> AnalysisExplanation:
        """
        Create explanation for why this result was detected.
        :param original_score: Score given by this recognizer
        :param explanation: Explanation string
        :return:
        """
        explanation = AnalysisExplanation(
            recognizer=self.__class__.__name__,
            original_score=original_score,
            textual_explanation=explanation,
        )
        return explanation

    @staticmethod
    def __check_label(
        entity: str, label: str, check_label_groups: Tuple[Set, Set]
    ) -> bool:
        return any(
            [entity in egrp and label in lgrp for egrp, lgrp in check_label_groups]
        )


# In[21]:




        # # Use Flair NER for identifying PII
        # sentence = Sentence(input_text)
        # tagger.predict(sentence)
        # entities = sentence.to_dict(tag_type='ner')['entities']
        
        # # Mask PII using Presidio analyzer
        # masked_text = analyzer.analyze(input_text, entities=entities)
        
    from flair.data import Sentence
    from flair.models import SequenceTagger

    def predict_ner_tags(input_text):
        

        # load tagger
        tagger = SequenceTagger.load("flair/ner-english-large")

        # make example sentence
        # sentence = Sentence("My name is Karishma Shirsath. I live in Toronto Canada.")

        sentence = Sentence(input_text)
        # predict NER tags
        tagger.predict(sentence)

        # print sentence
        print(sentence)

        # print predicted NER spans
        print("The following NER tags are found:")
        # iterate over entities and print
        for entity in sentence.get_spans("ner"):
            print(entity)



    # In[33]:

    
    def analyze_text(input_text):
        # load tagger
        tagger = SequenceTagger.load("flair/ner-english-large")

        # make example sentence
        sentence = Sentence(input_text)

        # predict NER tags
        tagger.predict(sentence)

        # print sentence
        print(sentence)

        # Anonymize identified named entities
        anonymized_sentence = str(sentence)
        for entity in sentence.get_spans("ner"):
            entity_text = entity.text
            anonymized_text = "*" * len(entity_text)
            anonymized_sentence = anonymized_sentence.replace(entity_text, anonymized_text)

        # print anonymized sentence
        print("Anonymized sentence:")
        print(anonymized_sentence)
        return anonymized_sentence