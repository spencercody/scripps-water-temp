from keras import Input, Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau






def create_lstm_model(input_shape)->Sequential:
    layers = [
        Input(shape=input_shape),
        LSTM(units=50, return_sequences=True),
        Dropout(.2),
        LSTM(units=50, return_sequences=False),
        Dropout(.2),
        Dense(1)
    ]

    lstm = Sequential(layers)

    return lstm

# ---------------------------------------------------------------------------- #

def get_callbacks():
    return [
        EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=.5,
            patience=2,
            min_lr=0.000001
        )
    ]
