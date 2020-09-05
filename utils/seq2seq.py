# %%
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

# %%


def get_model_1layer(encoder_input_size=1000, vocab_size=5000, embd_size=200, max_seq_len=50, lstm_units=500):

    # encoder
    enc_input = Input((None, encoder_input_size), name='enc_input')

    X = enc_input

    X = LSTM(lstm_units, return_state=True, name='enc_lstm')(X)

    _, enc_h, enc_c = X

    enc_states = [enc_h, enc_c]

    # decoder
    dec_input = Input((None,), name='dec_input')

    Y = dec_input

    Y = Embedding(vocab_size+1, embd_size, input_length=max_seq_len,
                  trainable=False, name='dec_embd', mask_zero=True)(Y)

    Y = LSTM(lstm_units, return_sequences=True, return_state=True,
             name='dec_lstm')(Y, initial_state=enc_states)

    dec_out, _, _ = Y

    Y = Dense(vocab_size, activation='softmax')(dec_out)

    # model
    model = Model(inputs=[enc_input, dec_input], outputs=Y)

    return model


# %%
model = get_model_1layer(2048)
model.summary()

# %%
