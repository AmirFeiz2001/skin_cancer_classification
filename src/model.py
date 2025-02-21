from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras import regularizers
from kerastuner.tuners import RandomSearch

def build_model(hp, num_features):
    # CNN branch for images
    model1_in = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(hp.Int('conv1_filters', 32, 128, step=32), (2, 2), padding='same', activation='relu')(model1_in)
    x = layers.Conv2D(hp.Int('conv2_filters', 32, 128, step=32), (2, 2), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(hp.Int('conv3_filters', 64, 256, step=64), (2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(hp.Int('conv4_filters', 64, 256, step=64), (2, 2), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(hp.Int('dense1_units', 128, 512, step=128), activation='relu')(x)
    model1_out = layers.Dense(2, activation='sigmoid')(x)

    # Simplified Dense branch for metadata
    model2_in = layers.Input(shape=(num_features,))
    x = layers.Dense(hp.Int('dense2_units', 64, 256, step=64), activation='relu')(model2_in)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1))(x)
    x = layers.Dense(hp.Int('dense3_units', 32, 128, step=32), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1))(x)
    model2_out = layers.Dense(2, activation='sigmoid')(x)

    # Merge branches
    concatenated = concatenate([model1_out, model2_out])
    x = layers.Dense(units=hp.Int('merge_units', 32, 256, step=32), activation='relu')(concatenated)
    out = layers.Dense(2, activation='sigmoid', name='output_layer',
                       kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)

    model = Model([model1_in, model2_in], out)
    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
