"""
Programa para guardar los datos atmosféricos y de contaminación en un documento
json con el siguiente formato:

{
    día1: {
        hora1: {
            PM10: number,
            PM25: number,
            ...
            temperatura: number
        },
        hora 2: {...}
    },
    día2: {...}
}

"""


import urllib.request
import datetime
import json
import csv

#Guardar los datos atmosféricos en un diccionario (leerlos de un csv descargado)
datos_atmosfericos = {}
with open(".\\datos_atmosfericos\\DH-5515X.csv") as file:
    read = csv.reader(file)
    for line in read:
        row = line[0].split(";")
        hour = datetime.datetime.strptime(row[5], "%Y-%m-%dT%H:%M:%S")
        datos_atmosfericos[hour.strftime("%Y/%m/%d %H:%M:%S")] = {
            "precipitacion":row[6],
            "velocidad_viento":row[11],
            "direccion_viento":row[14],
            "humedad":row[22],
            "temperatura":row[29]
        }


#Obtener de datos abiertos Junta de Andalucia los datos de contaminación del 25/11/2021 al 13/07/2023
data = {}
currentDate_datetime = datetime.datetime.strptime("20211025", '%Y%m%d').date()
end_date = datetime.datetime.strptime("20230713", '%Y%m%d').date()

while currentDate_datetime<=end_date:
    day_str_url = currentDate_datetime.strftime("%Y%m%d")
    day_str_dict = currentDate_datetime.strftime("%d/%m/%Y")
    url = f"https://www.juntadeandalucia.es/medioambiente/atmosfera/informes_siva/cuantitativo/{day_str_url[:4]}/GR_{day_str_url}.csv"

    try:
        response = urllib.request.urlopen(url)
    except urllib.error.HTTPError:
        currentDate_datetime = currentDate_datetime + datetime.timedelta(days=1)
        continue

    lines = [line.decode('utf-8') for line in response.readlines()]
    dataOfDay_csv = csv.reader(lines)
    data[day_str_dict] = {}

    for row in dataOfDay_csv:
        temp = row[0].replace(" ", "")
        hour_data = temp.split(";")
        
        if hour_data[2]=="GRANADA-NORTE":
            hour_atdict = (datetime.datetime.combine(currentDate_datetime,
                datetime.datetime.strptime(hour_data[4]+":00",
                "%H:%M:%S").time())).strftime("%Y/%m/%d %H:%M:%S")
            
            pollution = {
                "PM10": hour_data[5],
                "PM25":hour_data[6],
                "NO2":hour_data[7],
                "O3":hour_data[8],
                "SO2":hour_data[9]
            }

            try:
                atmosferic = datos_atmosfericos[hour_atdict]
            except KeyError:
                pass
            
            #Unir los datos de contaminación junto con los datos atmosféricos
            data[day_str_dict][hour_data[4]] = {**pollution, **atmosferic}
            
    
    currentDate_datetime = currentDate_datetime + datetime.timedelta(days=1)


#Guardar los datos en un documento json
with open("datos_norte.json", "w") as outfile:
    json.dump(data, outfile)
