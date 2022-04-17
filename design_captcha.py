import numpy as np
import io

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import model_from_json

class CTCLayer(layers.Layer):
    def __init__(self, name=None,**kwargs):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

# ORIGINAL
with open("design_model.json","r") as file:
  model_json = file.read()


loaded_model = model_from_json(model_json,custom_objects={"CTCLayer":CTCLayer()})
loaded_model.load_weights(("design_weights.h5"))

img_width = 200
img_height = 50
max_length = 5
AUTOTUNE = tf.data.experimental.AUTOTUNE
characters = ['j', '6', 'y', '7', 'r', 'h', 'u', 'n', 't', 'p', 'w', '0', 'f', 'm', 'v', 'q', 'c', '8', '3', 'o', 'b', 'd', 'e', '4', 'a', '1', '5', 'i', 'x', 'g', 'k', 'l', 's', '2']
prediction_model = keras.models.Model(
    loaded_model.get_layer(name="image").input, loaded_model.get_layer(name="dense2").output
)

char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        print(res)
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text
def encode_single_sample(img_path):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    # 7. Return a dict as our model is expecting two inputs
    return img

def design_captcha(path):
    title = ""
    img = (encode_single_sample(path))
    img = np.array([img])
    preds = prediction_model.predict(img)
    pred_texts = decode_batch_predictions(preds)
    for i in range(len(pred_texts)):
            title+= pred_texts[i]
    print(title)
    return title
