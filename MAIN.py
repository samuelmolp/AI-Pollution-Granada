"""
Algunas partes de este código fueron copidas y posteriormente modificadas de:
https://www.tensorflow.org/tutorials/structured_data/time_series
Tal y como figura en https://developers.google.com/terms/site-policies?hl=es-419,
este código se encuentra libre de uso según la licencia Apache 2.0
"""

import json
import numpy
import pandas
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt


class Window():
    """
    Clase para definir una ventana. Divide la información en ventanas
    y sirve para visualizar los resultados una vez ha sido entrenada la red neuronal
    """
    def __init__(self, input_width, shift, label_width, train_df, validation_df, test_df, label_column):

        #Parámetros de la longitud de la ventana y columnas
        self. input_width = input_width
        self.shift = shift 
        self.label_width = label_width
        self.total_width = self.input_width + shift

        self.label_column = label_column
        column_indices = {name: i for i, name in enumerate(train_df.columns)}
        self.column_index = column_indices[self.label_column]
    
        #Convertir los datasets en sliding windows
        self.train_df = self.make_dataset(train_df)
        self.validation_df = self.make_dataset(validation_df)
        self.test_df = self.make_dataset(test_df)

    def split_window(self, features):
        #Divide la informacion en inputs y labels (label solo con la columna que interesa)
        inputs = features[:, slice(0, self.input_width), :]
        labels = features[:, slice(self.total_width - self.label_width, None), :]

        labels = tf.stack([labels[:, :, self.column_index]], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        #Divide la información en sliding windows con batch size 32 y slide 1
        data = numpy.array(data, dtype=numpy.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_width, sequence_stride=1, shuffle=True, batch_size=32,)
        ds = ds.map(self.split_window) 
        
        return ds
    
    def plot(self, std, mean, models):
        #Visualizar las predicciones una vez han sido entrenadas las redes neuronales
        inputs, labels = next(iter(self.train_df))

        denormalize = lambda x: (x * std[self.label_column])+mean[self.label_column]
        inputs_deno = tf.map_fn(denormalize, inputs)
        labels_deno = tf.map_fn(denormalize, labels)

        plt.figure(figsize=(12, 8))

        input_indices = numpy.arange(self.total_width)[slice(0, self.input_width)]
        plt.plot(input_indices, inputs_deno[0, :, self.column_index],
                        label="Inputs", marker=".", zorder=-10, color="#00DDC2")
        
        label_indices = numpy.arange(self.total_width)[slice(self.total_width - self.label_width, None)]
        plt.scatter(label_indices, labels_deno[0, :,],
                        edgecolors="k", label="Labels", color="#00DDC2", s=32)
        
        markers = ["X", "^", "p"]
        colors = ["#FB9752", "#008374", "#2DCD5B"]
        for model in models:
            predictions = models[model](inputs)
            predictions = predictions[0, :, ] if model !="LSTM" else predictions[0, :, self.column_index]
            predictions = tf.map_fn(denormalize, predictions)

            plt.scatter(label_indices, predictions,
                    marker=markers.pop(0), edgecolors="k", label=model, s=40, color=colors.pop(0))
        
        plt.legend()
        plt.xlabel("Hora")
        plt.title(self.label_column)
        plt.show()


def getData(filename):
    """
    Esta función realiza las siguientes operaciones: 
    1. Leer la información del json y convertirla en un array de pandas
    2. Contabilizar el número de días sin lluvia
    3. Convertir la fecha en funciones trigonométricas
    4. Convertir el viento en un vector
    5. Dividir la información en 70% entrenamiento, 20% validación y 10% test
    6. Normalizar los valores
    7. Mostrar la gráfica de distribución de los valores

    Devuelve: sets de entrenamiento, validación y test, 
    array de la media de cada uno de los parámetros y de la desviación típica
    """
    with open(filename, "r") as openfile:
        data_json = json.load(openfile)

    data_array = []
    horas_sin_lluvia = -1 
    for day in data_json:
        for hour in data_json[day]:
            if hour =="MEDIA":
                continue
            temp = [f"{day} {hour}"]
            for e in data_json[day][hour]:
                if e=="O3":
                    continue

                temp.append(float(data_json[day][hour][e])) if data_json[day][hour][e]!="" else temp.append(0)
            
            horas_sin_lluvia=0 if temp[5]>0.0 else horas_sin_lluvia+1
            temp.append(horas_sin_lluvia/24)      

            data_array.append(temp)

    df = pandas.DataFrame(data_array, columns=["Time", "PM10", "PM25", "NO2", "SO2",
                "Precipitación", "vViento", "dViento", "Humedad", "Temperatura", "Dias sin lluvia"])
    
    date = pandas.to_datetime(df.pop("Time"), format="%d/%m/%Y %H:%M")
    timestamp_s = date.map(pandas.Timestamp.timestamp)
    day = 24*60*60
    year = (365.2425)*day
    df["Día seno"] = numpy.sin(timestamp_s * (2 * numpy.pi / day))
    df["Día coseno"] = numpy.cos(timestamp_s * (2 * numpy.pi / day))
    df["Año seno"] = numpy.sin(timestamp_s * (2 * numpy.pi / year))
    df["Año coseno"] = numpy.cos(timestamp_s * (2 * numpy.pi / year))

    wv = df.pop("vViento")
    wd_rad = df.pop("dViento")*numpy.pi / 180
    df["Wx"] = wv*numpy.cos(wd_rad)
    df["Wy"] = wv*numpy.sin(wd_rad)

    train_df = df[0:int(len(df)*0.7)]
    validation_df = df[int(len(df)*0.7):int(len(df)*0.9)]
    test_df = df[int(len(df)*0.9):]

    mean = train_df.mean()
    std = train_df.std()
    train_df = (train_df - mean) / std
    validation_df = (validation_df - mean) / std
    test_df = (test_df - mean) / std

    df_std = (df - mean) / std

    #Eliminar outliers para que se vea mejor
    for x in range(len(df_std["Precipitación"])): 
        if df_std["Precipitación"][x]>25:
            df_std["Precipitación"][x] = 25

    df_std = df_std.melt(var_name="Parámetro", value_name="Datos normalizados")
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x="Parámetro", y="Datos normalizados", data=df_std, bw=0.55)
    _ = ax.set_xticklabels(df.keys(), rotation=90)

    plt.show()

    return train_df, validation_df, test_df, mean, std


class Performance(tf.keras.callbacks.Callback):
    """
    Callback que se ejecuta al terminar cada epoch para añadir a la historia de entrenamiento 
    el error en el test 
    """
    def __init__(self, test):
        self.test = test

    def on_epoch_end(self, epoch, logs=None):
        logs["test_error"] = self.model.evaluate(self.test, verbose=0)[1]

        
def make_model(model, window, patience, epochs):
    """
    Dado un modelo y una ventana, compila y entrena el modelo
    Recibe también la paciencia y número de epochs
    Devuelve la historia de entrenamiento
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min")

    model.compile(loss=tf.keras.losses.MeanSquaredError(), #Error cuadrático medio como función de pérdida
                optimizer=tf.keras.optimizers.Adam(), #Optimizador Adam
                metrics=[tf.keras.metrics.MeanAbsoluteError()]) #Guardar el error absoluto medio (para las gráficas)

    history = model.fit(window.train_df, epochs=epochs,
                      validation_data=window.validation_df,
                      callbacks=[early_stopping, Performance(window.test_df)])
    
    return history


def epochsExperiment(models, pollutant):
    """
    Experimento utilizado para encontrar el número óptimo de epochs y paciencia. 
    Muestra una gráfica de la evolución del error con cada uno de los modelos 
    utilizando paciencia 2, 10 y 15. También muestra un gráfico de barras con el error final.

    Recibe: diccionario con los modelos y string del contaminante 
    """
    patiences = [2, 10, 15]
    error = {}

    for x in range(len(patiences)):
        error[patiences[x]] = {}

        train, validation, test, mean, std = getData("data_norte.json")
        window = Window(input_width=24, shift=24, label_width=24, 
                train_df=train, validation_df=validation, test_df=test,
                label_column=pollutant) 
        
        colors = ["#008374", "#00DDC2", "#7CFFEE"]
        for model in models:
            history = make_model(models[model], window, patiences[x], 500)
            error_plot_validation = history.history["val_mean_absolute_error"]
            error_plot_test = history.history["test_error"]
        
            colr = colors.pop(0)
            plt.plot(list(range(1, len(error_plot_validation)+1)), error_plot_validation, label=f"{model} validación", color=colr)
            plt.plot(list(range(1, len(error_plot_test)+1)), error_plot_test, label=f"{model} test", color=colr, linestyle="dotted")

            error[patiences[x]][model] = {
                "validation": models[model].evaluate(window.validation_df)[1],
                "test": models[model].evaluate(window.test_df, verbose=0)[1]
            }
        
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Error absoluto medio")
        plt.title(f"Evolución del error en {pollutant} con paciencia {patiences[x]}")
        plt.figure()

    plt.show()
    showError(error, "Paciencia", f"Error absoluto medio (validación)", "validation", "ERROR ABSOLUTO MEDIO EN VALIDACIÓN FRENTE A PACIENCIA")
    showError(error, "Paciencia", "Error absoluto medio (test)", "test", "ERROR ABSOLUTO MEDIO EN TEST FRENTE A PACIENCIA")
                 

def showError(error, labelx, labely, dataset, title):
    """
    Muestra una gráfica con el error absoluto medio medido de todos los modelos
    Flexible para un número de contaminantes indeterminado

    Recibe:
    -Diccionario error con la siguiente configuración:
    {
        contaminante 1: {
            modelo 1: {validation: error en set de validacion (float), test: error en set test (float)}
            modelo 2: {validation: float, test: float}
            modelo 3: {validation: float, test: float}
        },
        contaminante 2: {...},
        ...
    }

    - labelx, labely y title: strnigs con el título y nombre de los ejes
    - Dataset: "validation" o "test" para el error en el set de validación o test
    """

    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
    
    bar1 = numpy.arange(len(error))
    bar2 = [x + barWidth for x in bar1]
    bar3 = [x + barWidth for x in bar2]
    positions = [bar1, bar2, bar3]

    data_plot = [[] for x in range(3)]
    models = []

    for pollutant in error:
        x=0
        for model in error[pollutant]:
            models.append(model)
            data_plot[x].append(error[pollutant][model][dataset])
            x+=1

    colors = ["#008374", "#00DDC2", "#7CFFEE"]
    for x in range(3):
        plt.bar(positions[x], data_plot[x], width=barWidth, edgecolor="grey",
                label=models[x], color=colors.pop(0))
    
    
    plt.xlabel(labelx, fontweight ="bold", fontsize = 15)
    plt.ylabel(labely, fontweight ="bold", fontsize = 12)
    plt.xticks([r + barWidth for r in range(len(error))], error.keys())
    plt.title(title)
    plt.legend()
    plt.show()


class FeedBack(tf.keras.Model):
    """
    Define el modelo de RNN para predicciones de 24 horas.
    Cada prediccion se introduce como input al modelo
    """
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units

        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(14)


    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)

        return prediction, state
    
    def call(self, inputs, training=None):
        predictions = []
        prediction, state = self.warmup(inputs)

        predictions.append(prediction)

        for n in range(1, self.out_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)
            prediction = self.dense(x)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


def visualizeWeights(linear, train_df):
    """
    Visualizar los pesos de cada variable en la red neuronal lineal
    Recibe: modelo lineal entrenado y dataset de entrenamiento (sin dividir en ventanas)
    """
    plt.bar(x = range(len(train_df.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
    axis = plt.gca()
    axis.set_xticks(range(len(train_df.columns)))
    _ = axis.set_xticklabels(train_df.columns, rotation=90)

    plt.show()

if __name__ == "__main__":  
    getData("data_norte.json")
    
    pollutants = ["PM10", "PM25", "NO2", "SO2"]
    shift = 24 #1 para predicciones a una hora y 24 para predicciones a 24 horas

    error = {}
    for pollutant in pollutants:
        train, validation, test, mean, std = getData("data_norte.json")
        window = Window(input_width=24, shift=shift, label_width=24, 
                train_df=train, validation_df=validation, test_df=test,
                label_column=pollutant) 

        #Definición de la arquitectura de los modelos para la siguiente hora
        one_hour_models = {
            "Lineal": tf.keras.Sequential([tf.keras.layers.Dense(units=1)]),
            "Denso": 
            tf.keras.Sequential([
                tf.keras.layers.Dense(units=128, activation="relu"),
                tf.keras.layers.Dense(units=128, activation="relu"),
                tf.keras.layers.Dense(units=128, activation="relu"),
                tf.keras.layers.Dense(units=1)
            ]) if pollutant=="PM10" or pollutant=="PM25" else tf.keras.Sequential([
                tf.keras.layers.Dense(units=64, activation="relu"),
                tf.keras.layers.Dense(units=64, activation="relu"),
                tf.keras.layers.Dense(units=1)
            ]),
            "Recurrente": tf.keras.models.Sequential([
                tf.keras.layers.LSTM(32, return_sequences=True),
                tf.keras.layers.Dense(units=1)
            ])
        }
        
        #Definición de la arquitectura de los modelos para 24 horas
        multi_models = {
            "Lineal": tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                tf.keras.layers.Dense(24,kernel_initializer=tf.initializers.zeros()),
                tf.keras.layers.Reshape([24, 1])
            ]), 
            "Denso": tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                tf.keras.layers.Dense(1024, activation="relu"),
                tf.keras.layers.Dense(1024, activation="relu"),
                tf.keras.layers.Dense(24,kernel_initializer=tf.initializers.zeros()),
                tf.keras.layers.Reshape([24, 1])
            ]) if pollutant=="PM10" or pollutant=="PM25" else tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                tf.keras.layers.Dense(1024, activation="relu"),
                tf.keras.layers.Dense(24,kernel_initializer=tf.initializers.zeros()),
                tf.keras.layers.Reshape([24, 1])
            ]),
            "LSTM": FeedBack(32, 24)
        }
        
        models = one_hour_models if shift==1 else multi_models

        error[pollutant] = {}
        for model in models:
            history = make_model(models[model], window, 10, 30 if pollutant=="PM10" or pollutant=="PM25" else 10)
            error[pollutant][model] = {
                "validation": models[model].evaluate(window.validation_df)[1],
                "test": models[model].evaluate(window.test_df, verbose=0)[1]
            }

        window.plot(std, mean, models)
        if shift==1:
            visualizeWeights(models["Lineal"], train)
        

    print(error)
    showError(error,"Contaminante" ,"Error absoluto medio en el test", "test", "Error absoluto medio de cada modelo en cada contaminante")
    

