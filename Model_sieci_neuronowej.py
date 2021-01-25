""""
Po pobraniu dane zostały rozpakowane do folderu "Dane". Nazwa folderu z danymi treningowymi "stag1" została zmieniona na
"input". Następnie wszystkie podfoldery w tym folderze, opisane jako ID pacjentów, umieszczono w folderze "sample_data".
Dodatkowo do folderu "input" został przeniesiony plik z etykietami "stage1_labels.csv". Folder z danymi testowymi
"stage2" nie został wykorzystany, ponieważ nie była do niego dołączona lista etykiet pacjentów. Zamiast tego przykłady
sprawdzające działanie modelu wydzielono z folderu "sample_data".
Ostatecznie w folderze z danymi "Dane" znajdował się folder z danymi treningowymi "input", który zawierał plik z etykie-
tami "stage1_labels.csv" oraz podfolder "sample_data" ze skanami pacjentów.

Dane zostały pobrane ze strony: https://academictorrents.com/details/015f31a94c600256868be155358dc114157507fc
Program bazuje na kodzie ze strony: https://www.kaggle.com/zfturbo/keras-vs-cancer
"""

# Import bibliotek
import os
import cv2
import glob
import time
import random
import shutil
import numpy as np
import pandas as pd
import pydicom as dicom
from scipy import ndimage
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from keras.models import Sequential
np.random.seed(2016)

# Wersje najważnieszych bibliotek:
# Tensorflow: 2.1.0
# Keras: 2.2.4-tf
# Numpy: 1.19.1
print('Tensorflow: ', tf.__version__)
print('Keras: ', keras.__version__)
print('Numpy: ', np.__version__)

# Konfiguracja modelu:
conf = dict()

# Rozdzielczość skanów
conf['image_shape'] = (160, 160)
print('image_shape = ', conf['image_shape'])

# Parametry warstw
conf['level_1_filters'] = 4
print('level_1_filters = ', conf['level_1_filters'])
conf['level_2_filters'] = 8
print('level_2_filters = ', conf['level_2_filters'])
conf['dense_layer_size'] = 128
print('dense_layer_size = ', conf['dense_layer_size'])
conf['dropout_value'] = 0.5
print('dropout_value = ', conf['dropout_value'])

# Wskaźnik uczenia
conf['learning_rate'] = 1e-3
print('learning_rate = ', conf['learning_rate'])

# Rozmiar wsadu
conf['batch_size'] = 128
print('batch_size = ', conf['batch_size'])

# Liczba epok
conf['nb_epochs'] = 20
print('nb_epochs = ', conf['nb_epochs'])

# Liczba próbek treningowych na epokę, można ją ustawić opcjonalnie
conf['samples_train_per_epoch'] = 0
print('samples_train_per_epoch = ', conf['samples_train_per_epoch'])

# Liczba próbek walidacyjnych na epokę, można ją ustawić opcjonalnie
conf['samples_valid_per_epoch'] = 0
print('samples_valid_per_epoch = ', conf['samples_valid_per_epoch'])

# Granica podziału danych na zbiór testowy i walidacyjny
conf['train_valid_fraction'] = 0.7

# Opcja pozwalająca zapisać wagi modelu
conf['save_model'] = 1

# Katalog danych
data_dir = 'Dane'


# Utworzenia zboru testowego
def create_test_data(train_csv_table, data_size=20, move_back=False):
    new_stage2 = r'E:\{}\new_stage2'.format(data_dir)
    sample_images = r'E:\{}\new_stage2\sample_images'.format(data_dir)
    input_labels = r'E:\{}\input\stage1_labels_train.csv'.format(data_dir)
    new_stage_labels = r'E:\{}\new_stage2\new_stage2_labels.csv'.format(data_dir)
    new_train_csv_table = train_csv_table.head(len(train_csv_table.index)-data_size)
    valid_csv_table = train_csv_table.tail(data_size)

    for folder in (new_stage2, sample_images):
        if not os.path.exists(folder):
            os.mkdir(folder)

    new_train_csv_table.to_csv(input_labels, index=False)
    valid_csv_table.to_csv(new_stage_labels, index=False)

    for patient in valid_csv_table.id:
        if not os.path.exists(os.path.join(sample_images, patient)):
            src = r'E:\{}\input\sample_images\{}'.format(data_dir, patient)
            dst = sample_images
            shutil.move(src, dst)

    if move_back:
        for patient in os.listdir(sample_images):
            src = os.path.join(sample_images, patient)
            dst = r'E:\{}}\input\sample_images'.format(data_dir)
            shutil.move(src, dst)

    print('Files in train folder:', len(os.listdir(r'E:\{}\input\sample_images'.format(data_dir))))
    print('Files in validation folder:', len(os.listdir(sample_images)))


train_csv_table = pd.read_csv(r'E:\{}\input\stage1_labels.csv'.format(data_dir))
create_test_data(train_csv_table, 100, move_back=False)

train_csv_table = pd.read_csv(r'E:\{}\input\stage1_labels_train.csv'.format(data_dir))


# Wczytywanie i normalizacja skanów
def load_and_normalise_dicom(path, x, y):
    dicom1 = dicom.read_file(path)
    dicom_img = dicom1.pixel_array.astype(np.float64)
    dicom_img[dicom_img == -2000] = 0
    mn = dicom_img.min()
    mx = dicom_img.max()

    if (mx - mn) != 0:
        dicom_img = (dicom_img - mn)/(mx - mn)
    else:
        dicom_img[:, :] = 0
    if dicom_img.shape != (x, y):
        dicom_img = cv2.resize(dicom_img, (x, y), interpolation=cv2.INTER_CUBIC)

    return dicom_img


# Podział danych na zbiór treningowy i walidacyjny
def get_train_single_fold(train_data, fraction):
    ids = train_data['id'].values
    # random.shuffle(ids)
    split_point = int(round(fraction*len(ids)))
    train_list = ids[:split_point]
    valid_list = ids[split_point:]

    return train_list, valid_list


# Augmentacja danych
def augment(image, rescale_factor_range=(0.8, 1), rotation_angle_range=(-20, 20), shift=25, color_inverse=True,
            flip=True):
    height, width = image.shape
    if rescale_factor_range:
        if rescale_factor_range[0] > rescale_factor_range[1] or rescale_factor_range[0] < 0 or rescale_factor_range[1] \
                < 0:
            raise TypeError('invalid rescale factor shape')
        rescale_factor = np.random.random_sample() * (rescale_factor_range[1] - rescale_factor_range[0]) + \
                         rescale_factor_range[0]
        new_height = round(height * rescale_factor)
        new_width = round(height * rescale_factor)
        if rescale_factor < 1.0:
            img = np.zeros_like(image)
            row = (height - new_height) // 2
            col = (width - new_width) // 2
            img[row:row + new_height, col:col + new_width] = ndimage.zoom(image, (float(rescale_factor),
                                                                                  float(rescale_factor)),
                                                                          mode='nearest')[0:new_height, 0:new_width]
        elif rescale_factor > 1.0:
            row = (new_height - height) // 2
            col = (new_width - width) // 2
            img = ndimage.zoom(image[row:row + new_height, col:col + new_width], (float(rescale_factor),
                                                                                  float(rescale_factor)),
                               mode='nearest')
            extra_hight = (img.shape[0] - height) // 2
            extra_width = (img.shape[1] - width) // 2
            img = img[extra_hight:extra_hight + height, extra_width:extra_width + width]
        else:
            img = image
    else:
        img = image

    if rotation_angle_range:
        if rotation_angle_range[0] >= rotation_angle_range[1]:
            raise TypeError('invalid rotation angle factor shape')
        angel = np.random.random_sample() * (rotation_angle_range[1] - rotation_angle_range[0]) + rotation_angle_range[
            0]
        img = ndimage.rotate(img, angel, reshape=False)

    if shift:
        offset = np.array([[np.random.randint(-shift, shift)], [np.random.randint(-shift, shift)]])
        img = ndimage.interpolation.shift(img, (int(offset[0]), int(offset[1])), mode='nearest')

    if color_inverse:
        color_inverse_factor = np.random.randint(-1, 2)
        while color_inverse_factor == 0:
            color_inverse_factor = np.random.randint(-1, 2)
        img = img * color_inverse_factor

    if flip:
        flip_factor = np.random.randint(0, 2)
        if flip_factor:
            img = np.fliplr(img)
        else:
            img = np.flipud(img)
    return img


# Funkcja generująca dane treningowe
def batch_generator_train(files, train_csv_table, batch_size, do_aug=True):
    number_of_batches = np.ceil(len(files)/batch_size)
    counter = 0
    random.shuffle(files)
    while True:
        batch_files = files[batch_size*counter:batch_size*(counter+1)]
        image_list = []
        mask_list = []
        for f in batch_files:
            image = load_and_normalise_dicom(f, conf["image_shape"][0], conf["image_shape"][1])
            if do_aug:
                image = augment(image, rescale_factor_range=(0.8, 1), rotation_angle_range=(-20, 20), shift=25,
                                color_inverse=True, flip=True)
            patient_id = os.path.basename(os.path.dirname(f))
            is_cancer = train_csv_table.loc[train_csv_table['id'] == patient_id]['cancer'].values[0]
            if is_cancer == 0:
                mask = [0]
            else:
                mask = [1]
            image_list.append(image)
            mask_list.append(mask)
        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)
        image_list = np.expand_dims(image_list, axis=3)
        yield image_list, mask_list
        if counter == number_of_batches:
            random.shuffle(files)
            counter = 0


# Struktura sieci neuronowej
def CNN():
    model = Sequential()
    model.add(layers.Conv2D(filters=conf['level_1_filters'], kernel_size=(3, 3), activation='relu',
                            input_shape=(conf["image_shape"][0], conf["image_shape"][1], 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=conf['level_2_filters'], kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=conf['level_2_filters'], kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=conf['level_2_filters'], kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(conf['dense_layer_size'], activation='relu'))
    model.add(layers.Dropout(conf['dropout_value']))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=optimizers.RMSprop(lr=conf['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.SpecificityAtSensitivity(0.5)])

    return model


# Trenowanie modelu i wizualizacja wyników
def create_model_and_plots():
    train_patients, valid_patients = get_train_single_fold(train_csv_table, conf['train_valid_fraction'])
    print('Train patients: {}'.format(len(train_patients)))
    print('Valid patients: {}'.format(len(valid_patients)))

    train_files = []
    for patient in train_patients:
        train_files += glob.glob(r'E:\{}\input\sample_images\{}\*.dcm'.format(data_dir, patient))
    print('Number of train files: {}'.format(len(train_files)))

    valid_files = []
    for patient in valid_patients:
        valid_files += glob.glob(r'E:\{}\input\sample_images\{}\*.dcm'.format(data_dir, patient))
    print('Number of valid files: {}'.format(len(valid_files)))

    print('Create and compile model...')
    model = CNN()

    print('Fit model')
    steps_per_epoch = len(train_files) // conf['batch_size']
    validation_steps = len(valid_files) // conf['batch_size']
    if not conf['samples_valid_per_epoch'] == 0:
        steps_per_epoch = conf['samples_train_per_epoch']
        validation_steps = conf['samples_valid_per_epoch']
    if conf['samples_train_per_epoch'] < 0 or conf['samples_valid_per_epoch'] < 0:
        raise TypeError('invalid number of train samples per epoch')
    if conf['samples_train_per_epoch'] == 0 and conf['samples_valid_per_epoch'] != 0:
        raise TypeError('invalid number of validation samples per epoch')
    if conf['samples_train_per_epoch'] != 0 and conf['samples_valid_per_epoch'] == 0:
        raise TypeError('invalid number of train samples per epoch')
    print('Steps per epoch: {}, Validation steps: {}'.format(steps_per_epoch, validation_steps))
    start = time.time()
    history = model.fit_generator(generator=batch_generator_train(train_files, train_csv_table,
                                                                  conf['batch_size'], do_aug=True),
                                  validation_data=batch_generator_train(valid_files, train_csv_table,
                                                                        conf['batch_size'], do_aug=False),
                                  steps_per_epoch=steps_per_epoch,
                                  validation_steps=validation_steps,
                                  epochs=conf['nb_epochs'],
                                  verbose=1)
    end = time.time()
    print(end - start)
    plt.rcParams.update({'font.size': 15})
    hist = pd.DataFrame(history.history)
    epochs = history.epoch
    accuracy, val_accuracy, loss, val_loss = hist['specificity_at_sensitivity'], \
                                             hist['val_specificity_at_sensitivity'], hist['loss'], hist['val_loss']

    plt.subplot(211)
    plt.plot(epochs, accuracy, label='Czułość trenowania', marker='o')
    plt.plot(epochs, val_accuracy, label='Czułość walidacji', marker='o')
    plt.xlabel('Czułość')
    plt.ylabel('Epoki')
    plt.grid(True)
    plt.legend()

    plt.subplot(212)
    plt.plot(epochs, loss, label='Strata trenowania', marker='o')
    plt.plot(epochs, val_loss, label='Strata walidacji', marker='o')
    plt.xlabel('Strata')
    plt.ylabel('Epoki')
    plt.grid(True)
    plt.legend()
    plt.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=loss, name='Strata trenowania', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Strata walidacji', mode='markers+lines'))
    fig.update_layout(width=1600, height=800, xaxis_title='Epoki', yaxis_title='Strata',
                      font=dict(size=20, color="black"), legend=dict(font=dict(size=20, color="black")))
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=accuracy, name='Czułość trenowania', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=epochs, y=val_accuracy, name='Czułość walidacji', mode='markers+lines'))
    fig.update_layout(width=1600, height=800, xaxis_title='Epoki', yaxis_title='Czułość',
                      font=dict(size=20, color="black"), legend=dict(font=dict(size=20, color="black")))
    fig.show()

    return model


# Walidacja wyników końcowych
def create_submission_model(model):
    sample_subm = pd.read_csv(r'E:\{}\new_stage2\new_stage2_labels.csv'.format(data_dir))
    ids = sample_subm['id'].values
    for id in ids:
        print('Predict for patient: {}'.format(id))
        files = glob.glob(r'E:\{}\new_stage2\sample_images\{}\*.dcm'.format(data_dir, id))
        image_list = []
        for f in files:
            image = load_and_normalise_dicom(f, conf['image_shape'][0], conf['image_shape'][1])
            image_list.append(image)
        image_list = np.array(image_list)
        image_list = np.expand_dims(image_list, axis=3)
        batch_size = len(image_list)
        prediction = model.predict(image_list, verbose=1, batch_size=batch_size)
        print(prediction[:, 0])
        pred_value = prediction[:, 0].mean()
        sample_subm.loc[sample_subm['id'] == id, 'cancer'] = pred_value
    sample_subm.to_csv('subm.csv', index=False)


# Uruchomienie programu
if __name__ == '__main__':
    model = create_model_and_plots()
    if conf['save_model'] == 1:
        model.save('dsb.h5')
    create_submission_model(model)
