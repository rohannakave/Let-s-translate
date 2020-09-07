from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import europarl
import numpy as np
import math
import os
import tokenizer
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)

def sparse_cross_entropy(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean


def load():
    global sess
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    #set_session(sess)
    global model, model_encoder, model_decoder, data_src, data_dest, num_words, mark_start, mark_end, tokenizer_dest, tokenizer_src, language_code, token_start, token_end
    model_train = load_model('my_modeleng-fr.h5',  custom_objects={'sparse_cross_entropy':sparse_cross_entropy})
    model_encoder = load_model('model_encodereng_fr.hdf5')
    model_decoder = load_model('model_decodereng_fr.hdf5')

    global graph
    graph = tf.compat.v1.get_default_graph()
    language_code='fr'
    mark_start = 'ssss '
    mark_end = ' eeee'
    europarl.maybe_download_and_extract(language_code=language_code)

    data_src = europarl.load_data(english=True,
                          language_code=language_code)

    data_dest = europarl.load_data(english=False,
                           language_code=language_code,
                           start=mark_start,
                           end=mark_end)
    num_words = 10000
    tokenizer_src = tokenizer.TokenizerWrap(texts=data_src,
                           padding='pre',
                           reverse=True,
                           num_words=num_words)
    print("good")
    tokenizer_dest = tokenizer.TokenizerWrap(texts=data_dest,
                           padding='post',
                            reverse=False,
                            num_words=num_words)
    token_start = tokenizer_dest.word_index[mark_start.strip()]
    token_end = tokenizer_dest.word_index[mark_end.strip()]


@app.route('/', methods=['POST'])
def translate():
    data = request.get_json()
    input_text = data['sent']
    input_tokens = tokenizer_src.text_to_tokens(text=input_text, reverse=True, padding=True)
    with graph.as_default():
        tf.keras.backend.set_session(sess)
        #set_session(sess)
        initial_state = model_encoder.predict(input_tokens)
    max_tokens = tokenizer_dest.max_tokens
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)
    token_int = token_start
    output_text = ''
    count_tokens = 0
    with graph.as_default():
        tf.keras.backend.set_session(sess)
        #set_session(sess)
        while token_int != token_end and count_tokens < max_tokens:
            decoder_input_data[0, count_tokens] = token_int
            x_data = \
            {
                'decoder_initial_state': initial_state,
                'decoder_input': decoder_input_data
            }
            decoder_output = model_decoder.predict(x_data)
            token_onehot = decoder_output[0, count_tokens, :]
            token_int = np.argmax(token_onehot)
            sampled_word = tokenizer_dest.token_to_word(token_int)
            output_text += " " + sampled_word
            count_tokens += 1
    output_tokens = decoder_input_data[0]
    #print("Translated text:")
    return output_text.replace("eeee", "")
    


    #data = request.args()	
    #with graph.as_default():
        #Translated_text = translate(input_text=data)
    #return Traslated_text
    #return "hey"

if __name__ == "__main__":
    print("welcome")
    load()
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=8080)
