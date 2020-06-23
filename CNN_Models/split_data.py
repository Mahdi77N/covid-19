import json
import csv
from datetime import date
from datetime import timedelta
import datetime
import time
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler
from numpy import array

_CSV_Directory_ = ''
_JSON_Directory_ = ''

startDay = date.fromisoformat('2020-01-22')
endDay = date.fromisoformat('2020-05-08')
dayLen = (endDay - startDay).days
dataTrain = []
dataTest = []
hashCounties = [-1] * 78031     #78030 is biggest county fips

countiesData_temporal = {}
countiesData_fix = {}

input_shape = [0, 0, 0, 0]

def loadIntersection(jsonFilename):
    jsonMetaData = []
    with open(_JSON_Directory_ + jsonFilename) as jsonFile:
        jsonMetaData = json.load(jsonFile)
    return jsonMetaData

def loadCounties(csvFilename):
    csvData = []
    with open(_CSV_Directory_ + csvFilename) as csvFile:
        csvDriver = csv.DictReader(csvFile)
        for row in csvDriver:
            csvData.append(row)
    return csvData

# Compare two date. This function implemented for specefic use, getting baseDay object and counter, then compare it to targetDay in our Data format(e.g. 05/18/20)
def dayComp(baseDay, dayCounter, targetDay):
    day1 = (baseDay + timedelta(days=dayCounter)).isoformat()
    day2 = datetime.datetime.strptime(targetDay, '%m/%d/%y').strftime('%Y-%m-%d')
    if (day1 == day2):
        return True
    return False

def fromIsotoDataFormat(day):
    return day.strftime('%m/%d/%y')

def init_hashCounties():
    counties = loadCounties('full-data-county-fips.csv')
    for i in range(len(counties)):
        hashCounties[int(counties[i]['county_fips'], 10)] = i

def binary_search(target_fips, target_date):
    global countiesData_temporal
    target = (target_fips, date.fromisoformat(target_date))

    l = 0
    r = len(countiesData_temporal)

    # Find first row of target county
    while (1):
        mid = (r - l) // 2
        fips = int(countiesData_temporal[l + mid]['county_fips'], 10)

        if (fips == target[0] and l + mid > 0 and int(countiesData_temporal[l + mid - 1]['county_fips'], 10) != target[0]):
            l = l + mid
            r = l + 1000
            break

        elif (fips >= target[0]):
            r = l + mid

        else:
            l = l + mid + 1

        if (r == l):
            return -1

    target_daysFromStart = (target[1] - startDay).days
    if (target_daysFromStart <= dayLen):
        return l + target_daysFromStart
    else:
        return -1

def calculateIndex(target_fips, target_date):
    target = (target_fips, date.fromisoformat(target_date))

    target_daysFromStart = (target[1] - startDay).days
    target_countiesFromStart = hashCounties[target[0]]

    if (target_daysFromStart <= dayLen and target_countiesFromStart != -1):
        index = target_countiesFromStart * (dayLen + 1) + target_daysFromStart
        return (index, target_countiesFromStart)
    else:
        return (-1, target_countiesFromStart)

def calculateGridData(counties):
    global countiesData_temporal, countiesData_fix
    confirmed = 0
    death = 0
    virusPressure = 0
    virusPressure_weightSum = 0
    meat_plants = 0
    social_distancing_visitation_grade = 0
    social_distancing_visitation_grade_weightSum = 0
    population = 0
    area = 0
    population_density = 0
    longitude = 0
    longitude_sum = 0
    social_distancing_travel_distance_grade = 0
    social_distancing_travel_distance_grade_weightSum = 0
    houses = 0
    houses_density = 0
    for county in counties:
        index_temporal, index_fix = calculateIndex(county['fips'], (startDay + timedelta(days=i)).isoformat())
        if (index_temporal != -1):
            confirmed += round(float(countiesData_temporal[index_temporal]['confirmed']) * county['percent'])
            death += round(float(countiesData_temporal[index_temporal]['death']) * county['percent'])
            meat_plants += round(int(countiesData_fix[index_fix]['meat_plants'], 10) * county['percent'])
            virusPressure += float(countiesData_temporal[index_temporal]['virus-pressure']) * county['percent']
            virusPressure_weightSum += county['percent']
            social_distancing_visitation_grade += float(countiesData_temporal[index_temporal][ 'social-distancing-visitation-grade']) * county['percent']
            social_distancing_visitation_grade_weightSum += county['percent']
            population += round(int(countiesData_fix[index_fix]['total_population'], 10) * county['percent'])
            area += float(countiesData_fix[index_fix]['area']) * county['percent']
            longitude += float(countiesData_fix[index_fix]['longitude'])
            longitude_sum += 1
            social_distancing_travel_distance_grade += float(countiesData_temporal[index_temporal]['social-distancing-travel-distance-grade']) * county['percent']
            social_distancing_travel_distance_grade_weightSum += county['percent']
            houses += float(countiesData_fix[index_fix]['houses_density']) * float(countiesData_fix[index_fix]['area']) * county['percent']

    if virusPressure_weightSum != 0:
        virusPressure /= virusPressure_weightSum
    if social_distancing_visitation_grade_weightSum != 0:
        social_distancing_visitation_grade /= social_distancing_visitation_grade_weightSum
    if area != 0:
        population_density = round(population / area, 2)
        houses_density = round(houses / area, 2)
    if longitude_sum != 0:
        longitude /= longitude_sum
    if social_distancing_travel_distance_grade_weightSum != 0:
        social_distancing_travel_distance_grade /= social_distancing_travel_distance_grade_weightSum

    return [confirmed, round(virusPressure, 2), meat_plants, death, round(social_distancing_visitation_grade, 1), population_density, population, round(longitude, 3), round(social_distancing_travel_distance_grade, 1), houses_density]

def init_days():
    global startDay
    global endDay
    global dayLen
    startDay = date.fromisoformat(datetime.datetime.strptime(countiesData_temporal[0]['date'], '%m/%d/%y').strftime('%Y-%m-%d'))
    endDay = startDay
    
    for row in countiesData_temporal:
        day = date.fromisoformat(datetime.datetime.strptime(row['date'], '%m/%d/%y').strftime('%Y-%m-%d'))
        if day > endDay:
            endDay = day
            dayLen = (endDay - startDay).days

        elif day == startDay and row != countiesData_temporal[0]:
            break

if __name__ == "__main__":
    time_mainStart = time.time()

    gridIntersection = loadIntersection('map_intersection_1.json')
    countiesData_temporal = loadCounties('full-temporal-data.csv')
    countiesData_fix = loadCounties('full-fixed-data.csv')

    init_hashCounties()
    init_days()

    ################################################################ creating image array(CNN input) ### Binary Search

    time_imageCreation = time.time()
    
    # each row on imageArray include image data on day i
    imageArray = []

    for i in range(dayLen):
        grid = []
        for x in range(len(gridIntersection)):
            gridRow = []
            for y in range(len(gridIntersection[x])):
                gridCell = calculateGridData(gridIntersection[x][y])
                gridRow.append(gridCell)
            grid.append(gridRow)
        imageArray.append(grid)

    # # Show data
    # for i in range(len(imageArray)):
    #     print("day " + str(i))
    #     for x in range(len(imageArray[i])):
    #         for y in range(len(imageArray[i][x])):
    #             print(imageArray[i][x][y], end='')
    #         print('')
    #     print('')

    ################################################################ normalize data

    time_imageNormalization = time.time()

    imageNormal = []
    shape_imageArray = array(imageArray).shape
    
    normalizeObject = MinMaxScaler()
    imageNormal = normalizeObject.fit_transform(array(imageArray).reshape(shape_imageArray[0] * shape_imageArray[1] * shape_imageArray[2], shape_imageArray[3]))
    imageNormal = imageNormal.reshape(shape_imageArray[0], shape_imageArray[1], shape_imageArray[2], shape_imageArray[3])
        
    time_lap = time.time()

    # # Show data
    # for i in range(len(imageNormal)):
    #     print("day " + str(i))
    #     for x in range(len(imageNormal[i])):
    #         for y in range(len(imageNormal[i][x])):
    #             print(imageNormal[i][x][y], end='')
    #         print('')
    #     print('')

    ################################################################ split imageArray into train Data(dataTrain) and test Data(dataTest)

    # dataTrain = imageNormal[:-14]
    # dataTest = imageNormal[-28:]

    data_shape = (shape_imageArray[1], shape_imageArray[2], shape_imageArray[3])

    x_dataTrain = imageNormal[:-14][:-14]
    y_dataTrain = imageNormal[:-14][14:]

    x_dataTest = imageNormal[-28:][:-14]
    y_dataTest = imageNormal[-28:][14:]

    # Clear memory
    gridIntersection.clear()
    countiesData_temporal.clear()
    countiesData_fix.clear()
    imageArray.clear()
    # imageNormal.clear()

    ################################################################ print execution time
        
    time_endTime = time.time()

    print('\t|Image creation time: {0}'.format(time_imageNormalization - time_imageCreation))
    print('\t|Image normalization time: {0}'.format(time_lap - time_imageNormalization))
    print('\t|full execution time: {0}'.format(time_endTime - time_mainStart))

    ################################################################ init model
    model = keras.Sequential()
    # Conv2D parameters: filters, kernel_size, activation, input_shape
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=data_shape))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=data_shape))
    # model.add(tf.keras.layers.MaxPooling2D(2,2))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=data_shape))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=data_shape))
    # model.add(tf.keras.layers.MaxPooling2D(2,2))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=data_shape))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Dropout(0.25))
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # model.compile('adam', 'mean_squared_error')
    model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    model.fit(x_dataTrain, y_dataTrain, batch_size=32, epochs=10, verbose=1)
    score = model.evaluate(x_dataTest, y_dataTest, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
