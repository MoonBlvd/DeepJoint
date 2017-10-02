import tensorflow as tf

def softmax_loss(y_true, y_pred):
    """Compute softmax loss.

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).

    # Returns
        softmax_loss: Softmax loss, tensor of shape (?, num_boxes).
    """
    y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
    softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred),
                                  axis=-1)
    return softmax_loss