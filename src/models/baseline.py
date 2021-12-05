from tensorflow.keras.layers import (Conv2D, MaxPool2D, Dropout, Input, BatchNormalization,
                                    GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, ELU)
from tensorflow.keras.models import Model

def baseline_model(input_shape, filters, kernel_size, pool_size, dropout, global_pol, n_classes, verbose):

    inp = Input(shape=input_shape)

    for i in range(0, len(filters)):
        if i == 0:
            x = Conv2D(inp, filters[i], kernel_size)
        else:
            x = Conv2D(x, filters[i], kernel_size)
        
        x = BatchNormalization()(x)
        x = ELU()(x)
 
        x = MaxPool2D(pool_size=pool_size[i])(x)
        x = Dropout(rate=dropout[i])(x)

    if global_pol == 'avg':
         x = GlobalAveragePooling2D()(x)
    elif global_pol == 'max':
         x = GlobalMaxPooling2D()(x)

    x = Dense(units=n_classes, activation='softmax', name='pred_layer')(x)

    model = Model(inputs=inp, outputs=x)

    if verbose:
        print(model.summary())

    return model
