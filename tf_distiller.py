from transformers import *
import tensorflow_datasets
import tensorflow as tf

class DistilMyBertConfig(PretrainedConfig):
    def __init__(self,
                 vocab_size_or_config_json_file=33333,
                 num_classes=2,
                 distill_temperature=0.1,
                 task_balance=0.5,
                 max_seq_len=128,
                 epoch = 5,
                 **kwargs):
        super(PretrainedConfig, self).__init__(*kwargs)
        
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.num_classes=num_classes
            self.distill_temperature = distill_temperature
            self.task_balance = task_balance
            self.max_seq_len = max_seq_len
            self.epoch = epoch

# define knowledge distillation loss
def loss_fn(teacher_logits, student_logits, targets, config):
  one_hot_targets = tf.one_hot(targets, config.num_classes, dtype=tf.float32)
  loss_op_standard = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=student_logits, labels=one_hot_targets
  ))
  teacher_targets = tf.nn.softmax(tf.multiply(teacher_logits,  1.0 / config.distill_temperature))
  loss_op_soft = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=tf.multiply(student_logits, 1.0 / config.distill_temperature), labels=tf.stop_gradient(teacher_targets)
  ))
  # scale soft target obj to match hard target obj. scale
  loss_op_soft *= tf.square(config.distill_temperature)
  
  return loss_op_standard + loss_op_soft

# Get stuent and teacher nn architecture
student_config_class, student_model_class, student_tokenizer_class = DistilBertConfig, TFDistilBertForSequenceClassification, DistilBertTokenizer
teacher_config_class, teacher_model_class, teacher_tokenizer_class = BertConfig, TFBertForSequenceClassification, BertTokenizer

# Load student and teacher model
# Load teacher model from checkpoints, but freeze the teacher layers
teacher = teacher_model_class.from_pretrained("mrpc/1")#, output_hidden_states=True)
teacher_tokenizer = teacher_tokenizer_class.from_pretrained('mrpc/1') 
# Load student model from config
student_config_path = "distilbert-base-uncased.json"
stu_architecture_config = student_config_class.from_pretrained(student_config_path)
stu_architecture_config.sinusoidal_pos_embds = False
student = student_model_class(stu_architecture_config)

# Prepare training data
data = tensorflow_datasets.load('glue/mrpc')

train_dataset = glue_convert_examples_to_features(data['train'], teacher_tokenizer, max_length=128, task='mrpc')
valid_dataset = glue_convert_examples_to_features(data['validation'], teacher_tokenizer, max_length=128, task='mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
valid_dataset = valid_dataset.batch(64)

# prepare training
config = DistilMyBertConfig(distill_temperature=0.8, task_balance=0.5)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, epsilon=1e-06, clipnorm=5.0)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=student)
manager = tf.train.CheckpointManager(ckpt, '/tmp/distil', max_to_keep=5)

for x, y in train_dataset:
  with tf.GradientTape() as tape:
    teacher_logits = teacher(x['input_ids'])
    student_logits = student(x['input_ids'])
    targets = y
    loss = loss_fn(teacher_logits, student_logits, targets, config)
  # only optimize student weights
  gradients = tape.gradient(loss, student.trainable_variables)
  optimizer.apply_gradients(zip(gradients, student.trainable_variables))

  tf.saved_model.save(student, "mrpc/4")

train_loss = []
train_acc = []

for epoch in range(config.epoch):
  epoch_end_loss = tf.keras.metrics.Mean()
  epoch_end_acc = tf.keras.metrics.SparseCategoricalAccuracy()
  # Training loop - using batches of 16
  for step, (x, y) in enumerate(train_dataset):
    with tf.GradientTape() as tape:
      student_logits = student(x['input_ids'])[0]
      teacher_logits = teacher(x['input_ids'])[0]
      targets = y
      loss = loss_fn(teacher_logits, student_logits, targets, config)
    
    # only optimize student weights
    gradients = tape.gradient(loss, student.trainable_variables)
    optimizer.apply_gradients(zip(gradients, student.trainable_variables))

    # logging
    if step % 100 == 0:
      print(step, float(loss))
    
    # log accuracy
    pred = tf.nn.softmax(student_logits)
    epoch_end_acc(targets, pred)
    epoch_end_loss(loss)
    
  # End epoch
  train_acc.append(epoch_end_acc.result())
  train_loss.append(epoch_end_loss.result())
  
  # save checkpoint
  ckpt.step.assign_add(1)
  save_path = manager.save()
  print("Saved checkpoint for epoch {} at {}".format(int(ckpt.step), save_path))
  print("epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_end_loss.result(), epoch_end_acc.result()))
  

