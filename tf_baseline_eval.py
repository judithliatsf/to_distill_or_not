import tensorflow as tf
import time
from transformers import BertTokenizer, TFBertForSequenceClassification,DistilBertTokenizer, TFDistilBertForSequenceClassification, glue_convert_examples_to_features
import tensorflow_datasets

# Load MRPC data
data = tensorflow_datasets.load('glue/mrpc')

# Pick GPU device (only pick 1 GPU)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

# Load tokenizer, model from pretrained model/vocabulary
bert_tokenizer = BertTokenizer.from_pretrained('mrpc/1')
bert_model = TFBertForSequenceClassification.from_pretrained('mrpc/1')

valid_dataset = glue_convert_examples_to_features(data['validation'], bert_tokenizer, max_length=128, task='mrpc')
valid_dataset = valid_dataset.batch(64)

# Evaluate time for bert_model (bigger model)
start_time = time.time()
results = bert_model.predict(valid_dataset)
execution_time = time.time() - start_time

# Load tokenizer, model from pretrained model/vocabulary
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('mrpc/2')
distilbert_model = TFDistilBertForSequenceClassification.from_pretrained('mrpc/2')

valid_dataset = glue_convert_examples_to_features(data['validation'], distilbert_tokenizer, max_length=128, task='mrpc')
valid_dataset = valid_dataset.batch(64)

# Evaluate time for distilbert_model (bigger model)
start_time = time.time()
results = distilbert_model.predict(valid_dataset)
execution_time = time.time() - start_time
