# %%

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# %%

# %%
lstm_units = 500
encoder_input_size = 4096
time_steps_enc = 40
vocab_size = 5000
emd_size = 200
max_seq_len = 50

# %% Encoder

enc_input = Input((None, encoder_input_size), name='enc_input')
enc_lstm = LSTM(lstm_units, return_state=True, name='enc_lstm')
X = enc_input
X = enc_lstm(X)
enc_out, enc_h, enc_c = X
enc_states = [enc_h, enc_c]

# %%Decoder

dec_input = Input((None,), name='dec_input')
dec_embedding = Embedding(vocab_size+1, emd_size, input_length=max_seq_len,
                          trainable=False, name='dec_embd', mask_zero=True)
dec_lstm = LSTM(lstm_units, return_sequences=True,
                return_state=True, name='dec_lstm')
dec_dense = Dense(vocab_size, activation='softmax')
Y = dec_input
Y = dec_embedding(Y)
Y = dec_lstm(Y, initial_state=enc_states)
dec_out, _, _ = Y
Y = dec_dense(dec_out)

# %%Model

model = Model(inputs=[enc_input, dec_input], outputs=Y)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
# %%
model.summary()
plot_model(model, show_shapes=True)
# %%
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['acc'])
# %%
batch_size = 128
validation_split = 0.2
epochs = 20
validation_batch_size = 128

model.fit(x=[eng_padded], y=target_padded, batch_size=batch_size, epochs=epochs,
          validation_split=validation_split, validation_batch_size=validation_batch_size)

# %%

# %%inference model

encoder_model_inf = Model(enc_input, enc_states)

dec_inf_input_h = Input((lstm_units,))
dec_inf_input_c = Input((lstm_units,))
dec_inf_input_states = [dec_inf_input_h, dec_inf_input_c]

dec_inf_out, dec_inf_h, dec_inf_c = dec_lstm(dec_embedding(dec_input),
                                             initial_state=dec_inf_input_states)
dec_inf_states = [dec_inf_h, dec_inf_c]
dec_inf_output = dec_dense(dec_inf_out)

decoder_model_inf = Model(inputs=[dec_input] + dec_inf_input_states,
                          outputs=[dec_inf_output] + dec_inf_states)

# %%
encoder_model_inf.summary()
plot_model(encoder_model_inf, show_shapes=True)
# %%
decoder_model_inf.summary()
plot_model(decoder_model_inf, show_shapes=True)
# %%


def generate(input_seq):

    states_val = encoder_model_inf.predict(input_seq)
    target_seq = [word_index['\t']]

    predicted_sent = []
    stop_condition = False

    while not stop_condition:

        decoder_out, decoder_h, decoder_c = decoder_model_inf.predict(
            x=[np.array(target_seq)] + states_val)
        max_val_index = np.argmax(decoder_out[0, 0])

        if reverse_word_index[max_val_index] == '\n' or max_val_index == 0 or len(predicted_sent) > max_seq_len:
            stop_condition = True

        target_seq = [max_val_index]
        predicted_sent.append(reverse_word_index[max_val_index])
        states_val = [decoder_h, decoder_c]

    return " ".join(predicted_sent)

# %%


word_index = {"a": 1, "\t": 2}
reverse_word_index = {1: "a", 2: "\t", 4756: "HHH"}
generate(np.random.randn(1, 20, 4096))

# %%
