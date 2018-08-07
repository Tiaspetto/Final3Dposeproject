from data.data_scripts.readmat import *
from posenet_2d import *
import random
from keras.models import Model, load_model
import tensorflow as tf

def get_img_batch(batch_array):
    imgs = []
    for index in batch_array:
        img = read_image(index)
        imgs.append(img)

    return np.array(imgs)


def get_heat_batch(batch_array):
    heats = []
    for index in batch_array:
        heat = read_heat_info(index)
        heats.append(heat)

    return np.array(heats)


def get_train_batch(index_array, batch_size):

    while 1:
        for i in range(0, len(index_array), batch_size):
            batch_array = index_array[i:i+batch_size]
            x = get_img_batch(batch_array)
            y = get_heat_batch(batch_array)
            #print(np.shape(x), np.shape(y))
            yield (x,y)


def shuffle(index_array):
    for i in range(0, len(index_array)-1):
        index = random.randint(i, len(index_array)-1)
        index_array[index], index_array[i] = index_array[i], index_array[index]

    return index_array


if __name__ == '__main__':

    index_array = range(1, 2001)

    index_array = list(index_array)

    index_array = shuffle(index_array)

    ckpt_path = 'log/weights-{val_loss:.4f}.hdf5'
    ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min')

    model = ResNet50(input_shape=(224, 224, 3))
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['accuracy'])
    result = model.fit_generator(generator=get_train_batch(index_array, 8),
                                 steps_per_epoch=1351,
                                 callbacks=[ckpt],
                                 epochs=300, verbose=1,
                                 workers=1)
