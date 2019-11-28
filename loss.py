import torch.nn.functional as F
import torch
import tensorflow as tf

def loss_fct(teacher_logits, student_logits, targets, config):
  loss_op_hard = standard_loss_fct(student_logits, targets, config)
  loss_op_soft = ce_loss_fct(student_logits, teacher_logits, config)
  mse_loss = mse_loss_fct(teacher_logits, student_logits)
  return config.alpha_hard * loss_op_hard + config.alpha_soft * loss_op_soft + config.alpha_mse * mse_loss

def standard_loss_fct(student_logits, targets, config):
  one_hot_targets = tf.one_hot(targets, config.num_classes, dtype=tf.float32)
  loss_op_standard = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=student_logits, labels=one_hot_targets
  ))
  return loss_op_standard

def ce_loss_fct(student_logits, teacher_logits, config):
  teacher_targets = tf.nn.softmax(tf.multiply(teacher_logits,  1.0 / config.distill_temperature))
  loss_op_soft = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=tf.multiply(student_logits, 1.0 / config.distill_temperature), labels=tf.stop_gradient(teacher_targets)
  ))
  # scale soft target obj to match hard target obj. scale
  loss_op_soft *= tf.square(config.distill_temperature)
  return loss_op_soft

def ce_loss_fct_torch(student_logits, teacher_logits, config):
  loss = torch.nn.KLDivLoss(reduction='batchmean')
  return loss(F.log_softmax(student_logits/config.distill_temperature, dim=-1),
              F.softmax(teacher_logits/config.distill_temperature, dim=-1))*(config.distill_temperature)**2

def mse_loss_fct(teacher_logits, student_logits):
  return tf.reduce_mean(tf.keras.losses.MSE(teacher_logits, student_logits))

def mse_loss_fct_torch(teacher_logits, student_logits):
  return torch.nn.MSELoss(reduction='mean')(teacher_logits, student_logits)