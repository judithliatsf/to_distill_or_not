import tensorflow as tf
from transformers import *
import tensorflow_datasets

# Load MRPC dataset
data = tensorflow_datasets.load('glue/mrpc')

###################### BERT #########################

# Load tokenizer, model from pretrained model/vocabulary
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

train_dataset = glue_convert_examples_to_features(data['train'], bert_tokenizer, max_length=128, task='mrpc')
valid_dataset = glue_convert_examples_to_features(data['validation'], bert_tokenizer, max_length=128, task='mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
valid_dataset = valid_dataset.batch(64)

# Train full size bert-base-uncased

# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule 
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# Train and evaluate using tf.keras.Model.fit()
bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

history = bert_model.fit(train_dataset, epochs=2, steps_per_epoch=115,
                    validation_data=valid_dataset, validation_steps=7)

# Save model
bert_model.save_pretrained("mrpc/1/")
bert_tokenizer.save_pretrained("mrpc/1/")
###################### DistilBERT #########################

# Load tokenizer, model from pretrained model/vocabulary
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

train_dataset = glue_convert_examples_to_features(data['train'], distilbert_tokenizer, max_length=128, task='mrpc')
valid_dataset = glue_convert_examples_to_features(data['validation'], distilbert_tokenizer, max_length=128, task='mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
valid_dataset = valid_dataset.batch(64)

# Train distilled bert-base-uncased


# Train and evaluate using tf.keras.Model.fit()
distilbert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

history = distilbert_model.fit(train_dataset, epochs=2, steps_per_epoch=115,
                    validation_data=valid_dataset, validation_steps=7)

# Save
distilbert_model.save_pretrained("mrpc/2/")
distilbert_tokenizer.save_pretrained("mrpc/2/")