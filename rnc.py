
# RED NEURONAL CONVOLUCIONAL


# PRE PROCESAMIENTO DE IMAGENES

# Importar Paquetes Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten


# CREANDO EL MODELO

# Inicializando RNC
clasificador = Sequential()

# Parte 1 - Convolucion
clasificador.add(Conv2D(input_shape=(64,64,3),filters=32, kernel_size=3, strides=3, activation='relu'))

# Parte 2 - Agrupacion
clasificador.add(MaxPooling2D(pool_size=(2,2)))

# Parte 2.5 Capa Extra - Convolucion
clasificador.add(Conv2D(filters=32, kernel_size=3, strides=3, activation='relu'))
clasificador.add(MaxPooling2D(pool_size=(2,2)))

# Parte 3 - Aplanamiento
clasificador.add(Flatten())

# Parte 4 - Conexion Completa
clasificador.add(Dense(units=128, activation='relu'))
clasificador.add(Dense(units=1, activation='sigmoid'))

# Compilacion
clasificador.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])


# PARTE 2 - Encajando RNC en las imagenes
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

clasificador.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=10,
        validation_data=test_set,
        validation_steps=2000)





# Parte 3 - Haciendo una Prediccion

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/prediccion/gato.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
resultado = clasificador.predict(test_image)
training_set.class_indices

if resultado [0][0] == 1:
    prediccion = 'perro'
else:
    prediccion = 'gato'

















































