import tensorflow as tf


def cosine(q,a):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
    pooled_mul_12 = tf.reduce_sum(q * a, 1)
    score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8)
    return score

def CosineEmbeddingLoss(margin=0.):
  def _cosine_similarity(x1, x2):
    """Cosine similarity between two batches of vectors."""
    return tf.reduce_sum(x1 * x2, axis=-1) / (
        tf.norm(x1, axis=-1) * tf.norm(x2, axis=-1))
  def _cosine_embedding_loss_fn(input_one, input_two, target):
    similarity = _cosine_similarity(input_one, input_two)
    return tf.reduce_mean(tf.where(
        tf.equal(target, 1),
        1. - similarity,
        tf.maximum(tf.zeros_like(similarity), similarity - margin)))
  return _cosine_embedding_loss_fn