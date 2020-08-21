################################################################ Imports

import json
import csv
import sys
import datetime
import time
from datetime import date
from datetime import timedelta
from math import log2, floor

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, MaxPooling2D, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from numpy import array, zeros, save, load

import multiprocessing
from os import getpid

import email
import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import mimetypes
import os

################################################################ Defines

_CSV_Directory_ = ''
_JSON_Directory_ = ''
_INSTANCES_FILENAME_ = 'instances.npy'
_GRID_INTERSECTION_FILENAME_ = './map_intersection_square.json'
_COUNTIES_DATA_FIX_ = '../final data/full-fixed-data.csv'
_COUNTIES_DATA_TEMPORAL_ = '../final data/full-temporal-data.csv'
_CONUTIES_FIPS_ = './full-data-county-fips.csv'

_NO_PARALLEL_PROCESS_ = 8

################################################################ Globals

startDay = datetime.datetime.strptime('2020-01-22', '%Y-%m-%d')
endDay = datetime.datetime.strptime('2020-05-08', '%Y-%m-%d')
dayLen = (endDay - startDay).days
hashCounties = [-1] * 78031     #78030 is biggest county fips

countiesData_temporal = {}
countiesData_fix = {}

x_normalizers = []
y_normalizers = MinMaxScaler()

gridIntersection = []
countiesData_temporal = []
countiesData_fix = []

################################################################
# We change 5 parameters to find best model (for now, we can't change number of blocks(NO_blocks))
# input_size = [3, 5, 15, 25] where image size is 300*300
# hidden_dropout = [0, 0.2, 0.3, 0.4]
# visible_dropout = [0, 0.2, 0.3, 0.4]
# NO_dense_layer = [1, 2, 3]
# increase_filters = [0, 1]
################################################################

p1 = [3, 5, 15, 25]
p2 = [0, 0.2, 0.3, 0.4]
p3 = [0, 0.2, 0.3, 0.4]
p4 = [1, 2, 3]
p5 = [0, 1]

################################################################ Functions

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
    counties = loadCounties(_CONUTIES_FIPS_)
    for i in range(len(counties)):
        hashCounties[int(counties[i]['county_fips'], 10)] = i

def binary_search(target_fips, target_date):
    global countiesData_temporal
    target = (target_fips, datetime.datetime.strptime(target_date, '%Y-%m-%d'))

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
    target = (target_fips, datetime.datetime.strptime(target_date, '%Y-%m-%dT%H:%M:%S'))

    target_daysFromStart = (target[1] - startDay).days
    target_countiesFromStart = hashCounties[target[0]]

    if (target_daysFromStart <= dayLen and target_countiesFromStart != -1):
        index = target_countiesFromStart * (dayLen + 1) + target_daysFromStart
        return (index, target_countiesFromStart)
    else:
        return (-1, target_countiesFromStart)

def calculateGridData(counties):
    global countiesData_temporal, countiesData_fix
    death = 0
    confirmed = 0
    houses = 0
    houses_density = 0
    meat_plants = 0
    longitude = 0
    longitude_sum = 0
    social_distancing_travel_distance_grade = 0
    social_distancing_travel_distance_grade_weightSum = 0
    daily_state_test = 0
    daily_state_test_weightSum = 0
    population = 0
    passenger_load = 0
    population_density = 0
    area = 0
    for county in counties:
        index_temporal, index_fix = calculateIndex(county['fips'], (startDay + timedelta(days=i)).isoformat())
        if (index_temporal != -1):
            # sum
            death += round(float(countiesData_temporal[index_temporal]['death']) * county['percent'])
            confirmed += round(float(countiesData_temporal[index_temporal]['confirmed']) * county['percent'])
            passenger_load += round(float(countiesData_fix[index_fix]['passenger_load']) * county['percent'], 6)
            meat_plants += round(int(countiesData_fix[index_fix]['meat_plants'], 10) * county['percent'])
            population += round(int(countiesData_fix[index_fix]['total_population'], 10) * county['percent'])
            # average
            longitude += float(countiesData_fix[index_fix]['longitude'])
            longitude_sum += 1
            social_distancing_travel_distance_grade += float(countiesData_temporal[index_temporal]['social-distancing-travel-distance-grade']) * county['percent']
            social_distancing_travel_distance_grade_weightSum += county['percent']
            daily_state_test += float(countiesData_temporal[index_temporal]['daily-state-test']) * county['percent']
            daily_state_test_weightSum += county['percent']
            # density
            houses += float(countiesData_fix[index_fix]['houses_density']) * float(countiesData_fix[index_fix]['area']) * county['percent']
            area += float(countiesData_fix[index_fix]['area']) * county['percent']

    if daily_state_test_weightSum != 0:
        daily_state_test = round(daily_state_test / daily_state_test_weightSum, 2)
    if area != 0:
        population_density = round(population / area, 2)
        houses_density = round(houses / area, 2)
    if longitude_sum != 0:
        longitude = round(longitude / longitude_sum, 3)
    if social_distancing_travel_distance_grade_weightSum != 0:
        social_distancing_travel_distance_grade = round(social_distancing_travel_distance_grade / social_distancing_travel_distance_grade_weightSum, 1)

    output = []
    output.append(death)        #temporal
    output.append(confirmed)    #temporal
    output.append(houses_density)
    output.append(meat_plants)
    output.append(longitude)
    output.append(social_distancing_travel_distance_grade)  #temporal
    output.append(daily_state_test) #temporal
    output.append(population)
    output.append(passenger_load)
    output.append(population_density)
    return output

def init_days():
    global startDay
    global endDay
    global dayLen
    startDay = datetime.datetime.strptime(countiesData_temporal[0]['date'], '%m/%d/%y')
    endDay = startDay
    
    for row in countiesData_temporal:
        day = datetime.datetime.strptime(row['date'], '%m/%d/%y')
        if day > endDay:
            endDay = day
            dayLen = (endDay - startDay).days

        elif day == startDay and row != countiesData_temporal[0]:
            break

def split_d4Datas(imageArray, data_index):
    output = []
    for i in range(len(imageArray)):
        output.append([imageArray[i][data_index]])

    return output

# parse 28days data into 1 instance
def parse_data_into_instance(data):
    instance = []

    # add fixed data
    instance.append(data[0][2])
    instance.append(data[0][3])
    instance.append(data[0][4])
    instance.append(data[0][7])
    instance.append(data[0][8])
    instance.append(data[0][9])

    # add temporal data
    for i in range(14):
        instance.append(data[i][0])
        instance.append(data[i][1])
        instance.append(data[i][5])
        instance.append(data[i][6])

    result = data[27][0]

    return (instance, result)

def create_model(inputSize, hiddenDropout, visibleDropout, noBlocks, noDenseLayer, increaseFilters):
    noFilters = 64
    model = keras.Sequential()

    # Layers before first block
    model.add(tf.keras.layers.Conv2D(filters=noFilters, kernel_size = (3,3), padding='same', activation='relu', input_shape=(inputSize, inputSize, 62)))
    if (visibleDropout != 0):
        model.add(Dropout(visibleDropout))

    # layers in Blocks
    for i in range(noBlocks):
        if (increaseFilters == 1):
            noFilters = 64 * pow(2, i)
        model.add(Conv2D(filters=noFilters, kernel_size = (3,3), padding='same', activation="relu"))
        model.add(Conv2D(filters=noFilters, kernel_size = (3,3), padding='same', activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        if (hiddenDropout != 0):
            model.add(Dropout(hiddenDropout))

    # Layers after last block
    for i in range(noDenseLayer - 1):
        model.add(Dense(512,activation="relu"))
    # Last layer
    model.add(Dense(1,activation="relu"))

    model.compile('adam', 'mean_squared_error', metrics=['accuracy'])
    # model.compile(loss=keras.losses.poisson, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=tf.keras.losses.Poisson())
    return model

# This function expand the image, to get output size equal to input size
def pad_data(data, input_size):
    n = input_size // 2

    padded_data = list(data)
    for j in range(n):
        padded_data.insert(0, padded_data[0])
        padded_data.append(padded_data[-1])

    for i in range(len(padded_data)):
        padded_data[i] = list(padded_data[i])
        for j in range(n):
            padded_data[i].insert(0, padded_data[i][0])
            padded_data[i].append(padded_data[i][-1])
    return array(padded_data)

# This function extract windows with "input_size" size from image, train model with the windows data
def train_data(model, x_train, y_train, x_validation, y_validation, NO_epochs, input_size):
    data_shape = x_train.shape
    y_shape = y_train.shape
    
    padded_x = []
    padded_y = []

    for i in range(data_shape[0]):
        padded_x.append(pad_data(x_train[i], input_size))
        padded_y.append(pad_data(y_train[i], input_size))

    x_train = array(padded_x)
    y_train = array(padded_y)
    for i in range(data_shape[1]):
        for j in range(data_shape[2]):
            subX_trian = x_train[0:data_shape[0], i:i+input_size, j:j+input_size, 0:data_shape[3]]
            subY_train = y_train[0:data_shape[0], i:i+input_size, j:j+input_size, 0:y_shape[3]]

            subX_validation = x_validation[0:data_shape[0], i:i+input_size, j:j+input_size, 0:data_shape[3]]
            subY_validation = y_validation[0:data_shape[0], i:i+input_size, j:j+input_size, 0:y_shape[3]]

            model.fit(subX_trian, subY_train, batch_size=32, epochs=NO_epochs, verbose=1, validation_data=(subX_validation, subY_validation))

# This function extract windows with "input_size" size from image, evaluate model with the windows data
def evaluate_data(model, x_test, y_test, input_size):
    global x_normalizers
    data_shape = x_test.shape
    y_shape = y_test.shape
    y_test_org = y_normalizers.inverse_transform(y_test.reshape(y_shape[0] * y_shape[1] * y_shape[2], y_shape[3]))
    y_test_org = y_test_org.reshape(y_shape[0], y_shape[1], y_shape[2], y_shape[3])
    
    padded_x = []
    padded_y = []

    for i in range(data_shape[0]):
        padded_x.append(pad_data(x_test[i], input_size))
        padded_y.append(pad_data(y_test[i], input_size))

    x_test = array(padded_x)
    y_test = array(padded_y)

    sum_org = 0
    sum_MAE = 0
    sum_MAPE = 0
    sum_MASE = 0

    for i in range(data_shape[1]):
        for j in range(data_shape[2]):
            subX_test = x_test[0:data_shape[0], i:i+input_size, j:j+input_size, 0:data_shape[3]]

            subY_predict_normal = model.predict(subX_test)
            pred_shape = subY_predict_normal.shape
            subY_predict = y_normalizers.inverse_transform(subY_predict_normal.reshape(pred_shape[0] * pred_shape[1] * pred_shape[2], pred_shape[3]))
            subY_predict = subY_predict.reshape(pred_shape[0], pred_shape[1], pred_shape[2], pred_shape[3])

            for k in range(pred_shape[0]):
                sum_org += y_test_org[k][i][j][0]
                sum_MAE += abs(y_test_org[k][i][j][0] - subY_predict[k][0][0][0])
                sum_MAPE += abs(y_test_org[k][i][j][0] - subY_predict[k][0][0][0])
                sum_MASE += abs(y_test_org[k][i][j][0] - x_test[k][i][j][-4])

    MAE = sum_MAE / (data_shape[0] * data_shape[1] * data_shape[2])
    MAPE = (sum_MAPE / sum_org) / (data_shape[0] * data_shape[1] * data_shape[2])
    MASE = (sum_MASE / (data_shape[0] * data_shape[1] * data_shape[2])) / MAE

    return (MAE, MAPE, MASE)

# Use this function to log states of code, helps to find bugs
def log(str):
    t = datetime.datetime.now().isoformat()
    with open('log', 'a') as logFile:
        logFile.write('[{0}][{1}] {2}\n'.format(t, getpid(), str))

def save_process_result(process_number, parameters, result):
    t = datetime.datetime.now().isoformat()
    with open('process{0}.txt'.format(process_number), 'a') as resultFile:
        append_string = '[{0}][{1}]\n\t--model parameteres: {2}\n\t--result: MAE:{3}, MAPE:{4}, MASE:{5}\n'.format(t, getpid(), parameters, result[0], result[1], result[2])
        resultFile.write(append_string)

def send_result(process_number):
    filename = 'process{0}.txt'.format(process_number)
    send_email(filename)

# From prediction.py file
def send_email(*attachments):
    subject = "Server results"
    body = " "
    sender_email = "covidserver1@gmail.com"
    receiver_email = ["hadifazelinia78@gmail.com", "arezo.h1371@yahoo.com"]#
    CC_email = ["p.ramazi@gmail.com"]#
    password = "S.123456.S"

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = ','.join(receiver_email)#receiver_email
    message["Subject"] = subject
    message["CC"] = ','.join(CC_email) # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # Add attachments
    for file_name in attachments:
            f = open(file_name, 'rb')
            ctype, encoding = mimetypes.guess_type(file_name)
            if ctype is None or encoding is not None:
                ctype = 'application/octet-stream'
            maintype, subtype = ctype.split('/', 1)
            # in case of a text file
            if maintype == 'text':
                part = MIMEText(f.read(), _subtype=subtype)
            # any other file
            else:
                part = MIMEBase(maintype, subtype)
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(file_name))
            message.attach(part)
            f.close()
            text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email+CC_email , text)

################################################################ START

def create_instances():
    log('START: loading data form files')

    global gridIntersection, countiesData_temporal, countiesData_fix

    gridIntersection = loadIntersection(_GRID_INTERSECTION_FILENAME_)
    countiesData_temporal = loadCounties(_COUNTIES_DATA_TEMPORAL_)
    countiesData_fix = loadCounties(_COUNTIES_DATA_FIX_)

    init_hashCounties()
    init_days()

    ################################################################ creating image array(CNN input) ### Binary Search

    log('START: creating image')

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

    shape_imageArray = array(imageArray).shape
    imageArray = array(imageArray)

    ################################################################ creating instances

    log('START: creating instances')

    # 6fix data, 4temporal data, 4D: number of instances, datas, grid row, grid column
    instance_shape = (dayLen - 28, shape_imageArray[1], shape_imageArray[2], 14 * 4 + 6)
    x_instances = zeros(instance_shape)
    y_instances = zeros((dayLen - 28, shape_imageArray[1], shape_imageArray[2]))

    for i in range(dayLen - 28):
        for x in range(instance_shape[2]):
            for y in range(instance_shape[3]):
                features, result = parse_data_into_instance(imageArray[i:i+28, x, y, 0:10])
                for j in range(len(features)):
                    x_instances[i][x][y][j] = features[j]
                    y_instances[i][x][y] = result

    log('START: saving instances into disk')

    save('x_' + _INSTANCES_FILENAME_, x_instances)
    save('y_' + _INSTANCES_FILENAME_, y_instances)

################################################################ split imageArray into train, validation and test

def process_function(process_number, visible_dropout, NO_dense_layer, increase_filters):
    log('Process {1} started | parameters {0}'.format((visible_dropout, NO_dense_layer, increase_filters), process_number))

    x_instances = load('x_' + _INSTANCES_FILENAME_)
    y_instances = load('y_' + _INSTANCES_FILENAME_)
    instance_shape = x_instances.shape

    log('START: spliting data into train, validation and test')

    x_dataTrain = x_instances[:-42]
    y_dataTrain = y_instances[:-42]

    x_dataValidation = x_instances[-42:-21]
    y_dataValidation = y_instances[-42:-21]

    x_dataTest = x_instances[-21:]
    y_dataTest = y_instances[-21:]

    ################################################################ normalize data

    log('START: normalizing data')

    reshaped_x_dataTrain = x_dataTrain.reshape(x_dataTrain.shape[0] * instance_shape[1] * instance_shape[2], instance_shape[3])
    reshaped_y_dataTrain = y_dataTrain.reshape(y_dataTrain.shape[0] * instance_shape[1] * instance_shape[2], 1)
    reshaped_x_dataValidation = x_dataValidation.reshape(x_dataValidation.shape[0] * instance_shape[1] * instance_shape[2], instance_shape[3])
    reshaped_y_dataValidation = y_dataValidation.reshape(y_dataValidation.shape[0] * instance_shape[1] * instance_shape[2], 1)
    reshaped_x_dataTest = x_dataTest.reshape(x_dataTest.shape[0] * instance_shape[1] * instance_shape[2], instance_shape[3])
    reshaped_y_dataTest = y_dataTest.reshape(y_dataTest.shape[0] * instance_shape[1] * instance_shape[2], 1)

    normal_x_dataTrain = zeros((x_dataTrain.shape[0], instance_shape[1], instance_shape[2], instance_shape[3]))
    normal_x_dataValidation = zeros((x_dataValidation.shape[0], instance_shape[1], instance_shape[2], instance_shape[3]))
    normal_x_dataTest = zeros((x_dataTest.shape[0], instance_shape[1], instance_shape[2], instance_shape[3]))

    # Normal X_data
    for i in range(14*4 + 6):
        obj = MinMaxScaler()
        x_normalizers.append(obj)

        tempTrain = reshaped_x_dataTrain[:, i:i+1]
        tempTrain = obj.fit_transform(tempTrain)
        tempTrain = tempTrain.reshape(x_dataTrain.shape[0], instance_shape[1], instance_shape[2])

        tempValidation = reshaped_x_dataValidation[:, i:i+1]
        tempValidation = obj.transform(tempValidation)
        tempValidation = tempValidation.reshape(x_dataValidation.shape[0], instance_shape[1], instance_shape[2])

        tempTest = reshaped_x_dataTest[:, i:i+1]
        tempTest = obj.transform(tempTest)
        tempTest = tempTest.reshape(x_dataTest.shape[0], instance_shape[1], instance_shape[2])

        for j in range(instance_shape[0]):
            for k in range(instance_shape[1]):
                for s in range(instance_shape[2]):
                    if (j < x_dataTrain.shape[0]):
                        normal_x_dataTrain[j][k][s][i] = tempTrain[j][k][s]
                    if (j < x_dataValidation.shape[0]):
                        normal_x_dataValidation[j][k][s][i] = tempValidation[j][k][s]
                    if (j < x_dataTest.shape[0]):
                        normal_x_dataTest[j][k][s][i] = tempTest[j][k][s]

    # Normal Y_data
    normal_y_dataTrain = y_normalizers.fit_transform(reshaped_y_dataTrain)
    normal_y_dataTrain = normal_y_dataTrain.reshape(y_dataTrain.shape[0], instance_shape[1], instance_shape[2], 1)

    normal_y_dataValidation = y_normalizers.transform(reshaped_y_dataValidation)
    normal_y_dataValidation = normal_y_dataValidation.reshape(y_dataValidation.shape[0], instance_shape[1], instance_shape[2], 1)

    normal_y_dataTest = y_normalizers.transform(reshaped_y_dataTest)
    normal_y_dataTest = normal_y_dataTest.reshape(y_dataTest.shape[0], instance_shape[1], instance_shape[2], 1)

    ################################################################ evaluate_models

    log('START: Phase of testing models started')

    for i in range(len(p1)):
        for i2 in range(len(p2)):
            input_size = p1[i]
            hidden_dropout = p2[i2]

            log('Model testing with parameters {0}'.format((input_size, hidden_dropout, visible_dropout, NO_dense_layer, increase_filters)))

            NO_blocks = floor(log2(input_size))
            model = create_model(input_size, hidden_dropout, visible_dropout, NO_blocks, NO_dense_layer, increase_filters)
            train_data(model, normal_x_dataTrain, normal_y_dataTrain, normal_x_dataValidation, normal_y_dataValidation, 2, input_size)
            result = evaluate_data(model, normal_x_dataTest, normal_y_dataTest, input_size)

            log('result, MAE:{0}, MAPE:{1}, MASE:{2}'.format(result[0], result[1], result[2]))
            save_process_result(process_number, (input_size, hidden_dropout, visible_dropout, NO_dense_layer, increase_filters), result)

    log('Process {0} done'.format(process_number))
    try:
        send_result(process_number)
    except Exception as e:
        log('sending result via email failed')
        raise Exception(e)

################################################################ main

if __name__ == "__main__":
    processes = []
    parameters = []
    for i3 in range(len(p3)):
        for i4 in range(len(p4)):
            for i5 in range(len(p5)):
                parameters.append((p3[i3], p4[i4], p5[i5]))

    for i in range(len(parameters)):
        processes.append(multiprocessing.Process(target=process_function, args=(parameters[i][0], parameters[i][1], parameters[i][2], i, )))

    # Start parallel processes
    for i in range(_NO_PARALLEL_PROCESS_):
        log('Process number {0} starting'.format(i))
        processes[i].start()

    # Wait till 1 processes done, then start next one
    for i in range(len(processes) - 8):
        processes[i].join()
        processes[i + 8].start()

    # Wait for all processes done
    for i in range(_NO_PARALLEL_PROCESS_):
        processes[len(processes) - 8 + i].join()

    log('All processes done')
    try:
        send_email('log')
    except Exception as e:
        log('sending log file via email failed')
        raise Exception(e)
                        