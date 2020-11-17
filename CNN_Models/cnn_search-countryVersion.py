################################################################ Imports

import json
import csv
import sys
import datetime
import time
from datetime import date
from datetime import timedelta
from math import log2, floor, ceil, sqrt

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, MaxPooling2D, Dropout, AveragePooling2D

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from numpy import array, zeros, save, load, copyto

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
import progressbar

from numpy.random import seed
from tensorflow.random import set_seed

from normalizers import normal_x, normal_y, inverse_normal_y
from parameters import create_parameters

################################################################ Defines

_CSV_Directory_ = ''
_JSON_Directory_ = ''
_INSTANCES_FILENAME_ = 'instances.npy'
_GRID_INTERSECTION_FILENAME_ = './map_intersection_square.json'
_COUNTIES_DATA_FIX_ = '../csvFiles/full-fixed-data.csv'
_COUNTIES_DATA_TEMPORAL_ = '../csvFiles/full-temporal-data.csv'
_CONUTIES_FIPS_ = '../csvFiles/full-data-county-fips.csv'
_DISTRIBUTION_FILENAME_ = './distribution.json'
_RESULTS_DIR_ = 'results/'

_NO_PARALLEL_PROCESSES_ = 4
_NO_PROCESSES_ = 64
_NO_PROCESSES_Per_EMAIL_ = 16

_NUMPY_SEED_ = 580
_TENSORFLOW_SEED_ = 870

_PROGRESS_BAR_WIDGET_ = [progressbar.Percentage(), ' ', progressbar.Bar('=', '[', ']'), ' ']

################################################################ Log Function
# Use this function to log states of code, helps to find bugs
def log(str):
    t = datetime.datetime.now().isoformat()
    with open('log', 'a') as logFile:
        logFile.write('[{0}][{1}] {2}\n'.format(t, getpid(), str))

################################################################ Functions

def loadJsonFile(jsonFilename):
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

def calculateGridData(counties, i):
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
            death += (float(countiesData_temporal[index_temporal]['death']) * county['percent'])
            confirmed += (float(countiesData_temporal[index_temporal]['confirmed']) * county['percent'])
            passenger_load += (float(countiesData_fix[index_fix]['passenger_load']) * county['percent'])
            meat_plants += (int(countiesData_fix[index_fix]['meat_plants'], 10) * county['percent'])
            population += (int(countiesData_fix[index_fix]['total_population'], 10) * county['percent'])
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

    counties_dist = []
    for county in counties:
        index_temporal, index_fix = calculateIndex(county['fips'], (startDay + timedelta(days=i)).isoformat())
        if (index_temporal != -1):
            county_confirmed = (float(countiesData_temporal[index_temporal]['confirmed']) * county['percent'])
            if (county_confirmed != 0):
                counties_dist.append({'fips': county['fips'], 'percent': county_confirmed / confirmed})
            else:
                counties_dist.append({'fips': county['fips'], 'percent': 0})

    if daily_state_test_weightSum != 0:
        daily_state_test = (daily_state_test / daily_state_test_weightSum)
    if area != 0:
        population_density = (population / area)
        houses_density = (houses / area)
    if longitude_sum != 0:
        longitude = (longitude / longitude_sum)
    if social_distancing_travel_distance_grade_weightSum != 0:
        social_distancing_travel_distance_grade = (social_distancing_travel_distance_grade / social_distancing_travel_distance_grade_weightSum)

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
    return (output, counties_dist)

def init_days():
    global startDay
    global endDay
    global dayLen
    global countiesData_temporal
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

def create_model(inputSize, hiddenDropout, visibleDropout, noBlocks, noDenseLayer, increaseFilters, learning_rate, pooling_type, architecture_type):
    # Set random seeds to make situation equal for all models 
    seed(_NUMPY_SEED_)
    set_seed(_TENSORFLOW_SEED_)

    noFilters = 64
    model = keras.Sequential()

    # Layers before first block
    model.add(tf.keras.layers.Conv2D(filters=noFilters, kernel_size = (3,3), padding='same', input_shape=(inputSize, inputSize, 62)))
    if (visibleDropout != 1):
        model.add(Dropout(visibleDropout))

    # layers in Blocks
    for i in range(noBlocks):
        if (increaseFilters == 1):
            noFilters = 64 * pow(2, i)
        model.add(Conv2D(filters=noFilters, kernel_size = (3,3), padding='same'))
        model.add(Conv2D(filters=noFilters, kernel_size = (3,3), padding='same'))

        if (pooling_type == 'MaxPooling'):
            model.add(MaxPooling2D(pool_size=(2,2)))
        elif (pooling_type == 'AveragePooling'):
            model.add(AveragePooling2D(pool_size=(2,2)))
        else:
            log('Unknown Pooling type {0} | Default Pooling is using: MaxPooling'.format(pooling_type))
            model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(BatchNormalization())
        if (hiddenDropout != 1):
            model.add(Dropout(hiddenDropout))

    if (architecture_type == 1):
        model.add(Dense(1024))
        model.add(Dense(1024))
        model.add(Dense(1024))
        model.add(BatchNormalization())

    # Layers after last block
    for i in range(noDenseLayer - 1):
        model.add(Dense(1024))
    # Last layer
    model.add(Dense(1))

    model_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mape', optimizer=model_optimizer)
    return model

# This function expand the image, to get output size equal to input size
def pad_data(data, input_size):
    n = input_size // 2

    padded_data = list(data)
    for _ in range(n):
        padded_data.insert(0, padded_data[0])
        padded_data.append(padded_data[-1])

    for i in range(len(padded_data)):
        padded_data[i] = list(padded_data[i])
        for _ in range(n):
            padded_data[i].insert(0, padded_data[i][0])
            padded_data[i].append(padded_data[i][-1])
    return array(padded_data)

# This function extract windows with "input_size" size from image, train model with the windows data
def train_data(model, x_train, y_train, x_validation, y_validation, NO_epochs, input_size, batch_size):
    data_shape = x_train.shape
    y_shape = y_train.shape
    no_validation = x_validation.shape[0]
    
    padded_x = []
    padded_y = []

    for i in range(data_shape[0]):
        padded_x.append(pad_data(x_train[i], input_size))
        padded_y.append(pad_data(y_train[i], input_size))

    x_train = array(padded_x)
    y_train = array(padded_y)
    
    padded_x = []
    padded_y = []

    for i in range(no_validation):
        padded_x.append(pad_data(x_validation[i], input_size))
        padded_y.append(pad_data(y_validation[i], input_size))

    x_validation = array(padded_x)
    y_validation = array(padded_y)

    # clear memory
    del padded_x, padded_y

    for i in range(data_shape[1]):
        for j in range(data_shape[2]):
            subX_trian = x_train[0:data_shape[0], i:i+input_size, j:j+input_size, 0:data_shape[3]]
            subY_train = y_train[0:data_shape[0], i:i+input_size, j:j+input_size, 0:y_shape[3]]

            subX_validation = x_validation[0:data_shape[0], i:i+input_size, j:j+input_size, 0:data_shape[3]]
            subY_validation = y_validation[0:data_shape[0], i:i+input_size, j:j+input_size, 0:y_shape[3]]

            model.fit(subX_trian, subY_train, batch_size=batch_size, epochs=NO_epochs, verbose=0, validation_data=(subX_validation, subY_validation))

# This function extract windows with "input_size" size from image, evaluate model with the windows data
def evaluate_data(model, x_test, y_test, input_size, first_normal_param, second_normal_param):
    data_shape = x_test.shape
    y_test_org = inverse_normal_y(y_test, first_normal_param, second_normal_param)

    if (data_shape[0] != 21):
        log('datashape[0] supposed to be 21, but its {0}'.format(data_shape[0]))
    
    padded_x = []
    padded_y = []

    for i in range(data_shape[0]):
        padded_x.append(pad_data(x_test[i], input_size))
        padded_y.append(pad_data(y_test[i], input_size))

    x_test = array(padded_x)
    y_test = array(padded_y)

    # clear memory
    del padded_x, padded_y

    sum_org = 0
    sum_predict = 0
    sum_simple = 0
    sum_MAE = 0
    sum_MAPE = 0
    sum_MASE = 0

    sum_org_country = zeros(data_shape[0])
    sum_predict_country = zeros(data_shape[0])
    sum_simple_country = zeros(data_shape[0])
    
    sum_MAE_country = 0
    sum_MAPE_country = 0
    sum_MASE_country = 0

    for i in range(data_shape[1]):
        for j in range(data_shape[2]):
            subX_test = x_test[0:data_shape[0], i:i+input_size, j:j+input_size, 0:data_shape[3]]

            subY_predict_normal = model.predict(subX_test)
            pred_shape = subY_predict_normal.shape
            subY_predict = inverse_normal_y(subY_predict_normal, first_normal_param, second_normal_param)

            for k in range(pred_shape[0]):
                sum_org += y_test_org[k][i][j][0]
                sum_predict += subY_predict[k][0][0][0]
                sum_simple += x_test[k][i][j][-4]
                sum_MAE += abs(y_test_org[k][i][j][0] - subY_predict[k][0][0][0])
                sum_MAPE += abs(y_test_org[k][i][j][0] - subY_predict[k][0][0][0])
                sum_MASE += abs(y_test_org[k][i][j][0] - x_test[k][i][j][-4])
                
                sum_org_country[k] += y_test_org[k][i][j][0]
                sum_predict_country[k] += subY_predict[k][0][0][0]
                sum_simple_country[k] += x_test[k][i][j][-4]

    MAE_pixel = sum_MAE / (data_shape[0] * data_shape[1] * data_shape[2])
    MAPE_pixel = sum_MAPE / sum_org
    MASE_pixel = MAE_pixel / (sum_MASE / (data_shape[0] * data_shape[1] * data_shape[2]))
    
    # minus bias (min_prediction) from country predictions
    min_prediction = sum_predict_country[0]
    for i in range(data_shape[0]):
        if (sum_predict_country[i] < min_prediction):
            min_prediction = sum_predict_country[i]

    for i in range(data_shape[0]):
        sum_predict_country[i] -= min_prediction

    # calculating country errors
    for k in range(data_shape[0]):
        sum_MAE_country += abs(sum_org_country[k] - sum_predict_country[k])
        sum_MASE_country += abs(sum_org_country[k] - sum_simple_country[k])
        
        if (sum_org_country[k] != 0):
            sum_MAPE_country += abs(sum_org_country[k] - sum_predict_country[k]) / sum_org_country[k]
        else:
            log('sum_org_country[{0}] is zero'.format(k))

    MAE_country = sum_MAE_country / data_shape[0]
    MAPE_country = sum_MAPE_country / data_shape[0]
    MASE_country = MAE_country / (sum_MASE_country / data_shape[0])

    return (MAE_pixel, MAPE_pixel, MASE_pixel, MAE_country, MAPE_country, MASE_country)

def save_process_result(process_number, parameters, result):
    t = datetime.datetime.now().isoformat()
    with open(_RESULTS_DIR_ + 'process{0}.txt'.format(process_number), 'a') as resultFile:
        str_parameters = '[{0}][{1}]\n\t--model parameters: {2}\n\t'.format(t, getpid(), parameters)
        str_result_pixel = '--result for Pixels: MAE:{0}, MAPE:{1}, MASE:{2}\n\t'.format(result[0], result[1], result[2]) 
        str_result_country = '--result for Country, MAE:{0}, MAPE:{1}, MASE:{2}\n'.format(result[3], result[4], result[5])
        resultFile.write(str_parameters + str_result_pixel + str_result_country)

def save_best_result(process_number, parameters_pixel, result_pixel, parameters_country, result_country):
    with open(_RESULTS_DIR_ + 'process{0}.txt'.format(process_number), 'a') as resultFile:
        str_split = '========================================================\n'
        str_parameters_pixel = 'Best Pixel result\n\t--model parameters: {0}\n\t'.format(parameters_pixel)
        str_result_pixel = '--result for Pixels: MAE:{0}, MAPE:{1}, MASE:{2}\n\t'.format(result_pixel[0], result_pixel[1], result_pixel[2]) 
        str_parameters_country = 'Best Country result\n\t--model parameters: {0}\n\t'.format(parameters_country)
        str_result_country = '--result for Country, MAE:{0}, MAPE:{1}, MASE:{2}\n'.format(result_country[0], result_country[1], result_country[2])
        resultFile.write(str_split + str_parameters_pixel + str_result_pixel + str_parameters_country + str_result_country)

def save_process_result_ft(process_number, parameters, result):
    t = datetime.datetime.now().isoformat()
    with open(_RESULTS_DIR_ + 'process{0}_ft.txt'.format(process_number), 'a') as resultFile:
        str_parameters = '[{0}][{1}]\n\t--model parameters: {2}\n\t'.format(t, getpid(), parameters)
        str_result_pixel = '--result for Pixels: MAE:{0}, MAPE:{1}, MASE:{2}\n\t'.format(result[0], result[1], result[2]) 
        str_result_country = '--result for Country, MAE:{0}, MAPE:{1}, MASE:{2}\n'.format(result[3], result[4], result[5])
        resultFile.write(str_parameters + str_result_pixel + str_result_country)

def save_best_result_ft(process_number, parameters_pixel, result_pixel, parameters_country, result_country):
    with open(_RESULTS_DIR_ + 'process{0}_ft.txt'.format(process_number), 'a') as resultFile:
        str_split = '========================================================\n'
        str_parameters_pixel = 'Best Pixel result\n\t--model parameters: {0}\n\t'.format(parameters_pixel)
        str_result_pixel = '--result for Pixels: MAE:{0}, MAPE:{1}, MASE:{2}\n\t'.format(result_pixel[0], result_pixel[1], result_pixel[2]) 
        str_parameters_country = 'Best Country result\n\t--model parameters: {0}\n\t'.format(parameters_country)
        str_result_country = '--result for Country, MAE:{0}, MAPE:{1}, MASE:{2}\n'.format(result_country[0], result_country[1], result_country[2])
        resultFile.write(str_split + str_parameters_pixel + str_result_pixel + str_parameters_country + str_result_country)

# From prediction.py file
def send_email(*attachments):
    subject = "Server results"
    body = " "
    sender_email = "covidserver1@gmail.com"
    receiver_email = ["hadifazelinia78@gmail.com", "arezo.h1371@yahoo.com"]#
    CC_email = ["p.ramazi@gmail.com", "zeinab.maleki@gmail.com"]#
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

def send_private_email(*attachments):
    subject = "Server results"
    body = " "
    sender_email = "covidserver1@gmail.com"
    receiver_email = ["hadifazelinia78@gmail.com"]#
    CC_email = []#
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

def send_result(process_numbers):
    files = []
    for process_number in process_numbers:
        filename = _RESULTS_DIR_ + 'process{0}_ft.txt'.format(process_number)
        files.append(filename)
    try:
        send_email(files)
    except Exception as e:
        log('sending result of processes {0} via email failed: {1}'.format(process_numbers, e))

def send_private_result(process_number):
    try:
        process_filename = _RESULTS_DIR_ + 'process{0}.txt'.format(process_number)
        process_ft_filename = _RESULTS_DIR_ + 'process{0}_ft.txt'.format(process_number)
        log_filename = 'log'
        send_private_email(process_filename, process_ft_filename, log_filename)
    except Exception as e:
        log('sending result of processes {0} via email failed: {1}'.format(process_number, e))

def send_log():
    try:
        send_private_email('log')
    except Exception as e:
        log('sending log file via email failed')
        raise Exception(e)

# This function init number of parallel processes with number of cpu
def init_no_processes():
    global _NO_PARALLEL_PROCESSES_
    _NO_PARALLEL_PROCESSES_ = multiprocessing.cpu_count()
    # limit no processes to 20
    if (_NO_PARALLEL_PROCESSES_ > 22):
        log('cpu core counts: {0}'.format(_NO_PARALLEL_PROCESSES_))
        _NO_PARALLEL_PROCESSES_ = 22
        
    log('_NO_PARALLEL_PROCESSES_ set to {0}'.format(_NO_PARALLEL_PROCESSES_))

def save_last_process(process_number):
    with open('last_process.txt', 'w') as fd:
        fd.write(str(process_number))

def calculate_county_error(test_start_day, predictions):
    init_hashCounties()
    init_days()
    countiesData_temporal = loadCounties(_COUNTIES_DATA_TEMPORAL_)

    sum_org = 0
    sum_predict = 0
    sum_MAE = 0
    sum_MASE = 0

    counties = loadCounties(_CONUTIES_FIPS_)
    for i in range(len(counties)):
        fips = int(counties[i]['county_fips'], 10)
        index_temporal, index_fix = calculateIndex(fips, (startDay + timedelta(days=test_start_day)).isoformat())
        if (index_temporal != -1):
            for k in range(21):
                orginal_death = float(countiesData_temporal[index_temporal + k]['death'])
                prediction_death = predictions[k][fips]
                simple_death = float(countiesData_temporal[index_temporal + k - 14]['death'])
                
                sum_org += orginal_death
                sum_predict += prediction_death
                sum_MAE += abs(orginal_death - prediction_death)
                sum_MASE += abs(orginal_death - simple_death)
        else:
            log('index = -1 | startDay={0}, fips={1}, index_fix={2}, test_start_day={3}'.format(startDay, fips, index_fix, test_start_day))

    MAE = sum_MAE / (21 * len(counties))
    MAPE = sum_MAE / sum_org
    MASE = MAE / (sum_MASE / (21 * len(counties)))

    return (MAE, MAPE, MASE)


################################################################ Globals

startDay = datetime.datetime.strptime('2020-01-22', '%Y-%m-%d')
endDay = datetime.datetime.strptime('2020-05-08', '%Y-%m-%d')
dayLen = (endDay - startDay).days
hashCounties = [-1] * 78031     #78030 is biggest county fips

countiesData_temporal = {}
countiesData_fix = {}

gridIntersection = loadJsonFile(_GRID_INTERSECTION_FILENAME_)
countiesData_temporal = loadCounties(_COUNTIES_DATA_TEMPORAL_)
countiesData_fix = loadCounties(_COUNTIES_DATA_FIX_)

################################################################ START

def create_instances():
    global gridIntersection, countiesData_temporal, countiesData_fix

    ################################################################ creating image array(CNN input) ### Binary Search

    log('creating image')

    # each row on imageArray include image data on day i
    imageArray = []
    distribution = []

    for i in range(dayLen):
        grid = []
        grid_distribution = []
        for x in range(len(gridIntersection)):
            gridRow = []
            gridRow_distribution = []
            for y in range(len(gridIntersection[x])):
                gridCell, gridCell_distribution = calculateGridData(gridIntersection[x][y], i)
                gridRow.append(gridCell)
                gridRow_distribution.append(gridCell_distribution)
            grid.append(gridRow)
            grid_distribution.append(gridRow_distribution)
        imageArray.append(grid)
        distribution.append(grid_distribution)

    with open(_DISTRIBUTION_FILENAME_, 'w') as fd:
        fd.write(json.dumps(distribution))
        log('distribution file created')

    shape_imageArray = array(imageArray).shape
    imageArray = array(imageArray)

    ################################################################ creating instances

    log('creating instances')

    # 6fix data, 4temporal data, 4D: number of instances, datas, grid row, grid column
    instance_shape = (dayLen - 28, shape_imageArray[1], shape_imageArray[2], 14 * 4 + 6)
    x_instances = zeros(instance_shape)
    y_instances = zeros((dayLen - 28, shape_imageArray[1], shape_imageArray[2]))

    for i in range(dayLen - 28):
        for x in range(instance_shape[1]):
            for y in range(instance_shape[2]):
                features, result = parse_data_into_instance(imageArray[i:i+28, x, y, 0:10])
                for j in range(len(features)):
                    x_instances[i][x][y][j] = features[j]
                    y_instances[i][x][y] = result

    log('saving instances into disk')

    save('x_' + _INSTANCES_FILENAME_, x_instances)
    save('y_' + _INSTANCES_FILENAME_, y_instances)

    # Clear memory
    del x_instances
    del y_instances
    del gridIntersection
    del countiesData_temporal
    del countiesData_fix
    del imageArray

################################################################ evaluate_models

def process_function(parameters, 
            process_number, 
            start, 
            end, 
            shared_x_train, 
            shared_y_train, 
            shared_x_validation, 
            shared_y_validation, 
            shared_x_final_test, 
            shared_y_final_test,
            first_normal_param,
            second_normal_param, ):
    log('Process {1} started | parameters {0}'.format((start, end), process_number))

    pixel_best_model = -1
    pixel_best_result = (-1, -1, -1)

    country_best_model = -1
    country_best_result = (-1, -1, -1)

    pixel_best_model_ft = -1
    pixel_best_result_ft = (-1, -1, -1)

    country_best_model_ft = -1
    country_best_result_ft = (-1, -1, -1)

    for i in range(start, end):
        input_size = parameters[i][0]
        hidden_dropout = parameters[i][1] 
        visible_dropout = parameters[i][2] 
        NO_dense_layer = parameters[i][3]
        increase_filters = parameters[i][4]
        learning_rate = parameters[i][5]
        batch_size = parameters[i][6]
        pooling_type = parameters[i][7]
        architecture_type = parameters[i][8]

        log('Model testing with parameters {0}'.format((input_size, hidden_dropout, visible_dropout, NO_dense_layer, increase_filters, learning_rate, batch_size, pooling_type, architecture_type)))
        NO_blocks = floor(log2(input_size))
        model = create_model(input_size, hidden_dropout, visible_dropout, NO_blocks, NO_dense_layer, increase_filters, learning_rate, pooling_type, architecture_type)
        train_data(model, shared_x_train, shared_y_train, shared_x_validation, shared_y_validation, 2, input_size, batch_size)
        result = evaluate_data(model, shared_x_validation, shared_y_validation, input_size, first_normal_param, second_normal_param)

        log('result for Pixels, MAE:{0}, MAPE:{1}, MASE:{2}'.format(result[0], result[1], result[2]))
        log('result for Country, MAE:{0}, MAPE:{1}, MASE:{2}'.format(result[3], result[4], result[5]))
        save_process_result(process_number, parameters[i], result)

        if (pixel_best_model == -1 or result[2] < pixel_best_result[2]):
            pixel_best_model = i
            pixel_best_result = (result[0], result[1], result[2])

        if (country_best_model == -1 or result[5] < country_best_result[2]):
            country_best_model = i
            country_best_result = (result[3], result[4], result[5])

        result = evaluate_data(model, shared_x_final_test, shared_y_final_test, input_size, first_normal_param, second_normal_param)
        save_process_result_ft(process_number, parameters[i], result)

        if (pixel_best_model_ft == -1 or result[2] < pixel_best_result_ft[2]):
            pixel_best_model_ft = i
            pixel_best_result_ft = (result[0], result[1], result[2])

        if (country_best_model_ft == -1 or result[5] < country_best_result_ft[2]):
            country_best_model_ft = i
            country_best_result_ft = (result[3], result[4], result[5])

    log('Process {0} done'.format(process_number))
    save_best_result(process_number, parameters[pixel_best_model], pixel_best_result, parameters[country_best_model], country_best_result)
    save_best_result_ft(process_number, parameters[pixel_best_model_ft], pixel_best_result_ft, parameters[country_best_model_ft], country_best_result_ft)

################################################################ main

if __name__ == "__main__":
    log('loading data form files')
    init_no_processes()
    init_hashCounties()
    init_days()

    # Check if instances are ready
    if (os.path.exists('x_' + _INSTANCES_FILENAME_) and os.path.exists('y_' + _INSTANCES_FILENAME_) and os.path.exists(_DISTRIBUTION_FILENAME_)):
        log('instances found')
    else:
        log('creating instances')
        create_instances()

    x_instances = load('x_' + _INSTANCES_FILENAME_)
    y_instances = load('y_' + _INSTANCES_FILENAME_)

    ################################################################ split imageArray into train, validation and test

    log('spliting data into train, validation and test')

    x_dataTrain = x_instances[:-42]
    y_dataTrain = y_instances[:-42]

    x_dataValidation = x_instances[-42:-21]
    y_dataValidation = y_instances[-42:-21]

    x_dataFinalTest = x_instances[-21:]
    y_dataFinalTest = y_instances[-21:]

    ################################################################ normalize data

    log('normalizing data')

    normal_x_dataTrain, normal_x_dataValidation, normal_x_dataFinalTest = normal_x(x_dataTrain, x_dataValidation, x_dataFinalTest)
    normal_y_dataTrain, normal_y_dataValidation, normal_y_dataFinalTest, first_normal_param, second_normal_param = normal_y(y_dataTrain, y_dataValidation, y_dataFinalTest)

    ################################################################ clearing memory

    del x_instances, x_dataTrain, x_dataValidation, x_dataFinalTest
    del y_instances, y_dataTrain, y_dataValidation, y_dataFinalTest

    log('Phase of testing models started')

    print('[*] Number of parallel processes: {0}'.format(_NO_PARALLEL_PROCESSES_))

    ################################################################ creating parameters and results directory

    parameters = create_parameters()

    # create results directory if it's missing
    if (os.path.exists(_RESULTS_DIR_) == False):
        os.mkdir(_RESULTS_DIR_)

    ################################################################ creating processes

    processes = []
    model_per_process = ceil(len(parameters) / _NO_PROCESSES_)

    for i in range(_NO_PROCESSES_):
        processes.append(multiprocessing.Process(target=process_function, args=(
            parameters, 
            i, 
            i * model_per_process, 
            min((i + 1) * model_per_process, len(parameters)), 
            normal_x_dataTrain, 
            normal_y_dataTrain, 
            normal_x_dataValidation, 
            normal_y_dataValidation, 
            normal_x_dataFinalTest, 
            normal_y_dataFinalTest,
            first_normal_param,
            second_normal_param, )))

    start_process = 0
    try:
        with open('last_process.txt', 'r') as fd:
            start_process = int(fd.read(), 10) + 1
    except:
        start_process = 0

    # progressbar
    progressBar = progressbar.ProgressBar(maxval=_NO_PROCESSES_, widgets=_PROGRESS_BAR_WIDGET_)
    progressBar.start()

    progressBar.update(start_process)

    # Start parallel processes
    for i in range(_NO_PARALLEL_PROCESSES_):
        log('Process number {0} starting'.format(i + start_process))
        processes[i + start_process].start()

    # Wait till 1 process done, then start the next one
    for i in range(_NO_PROCESSES_ - start_process - _NO_PARALLEL_PROCESSES_):
        processes[i + start_process].join()
        progressBar.update(i + start_process)
        send_private_result(i + start_process)
        save_last_process(i + start_process)
        processes[i + start_process + _NO_PARALLEL_PROCESSES_].start()

        if ((i + start_process) % _NO_PROCESSES_Per_EMAIL_ == 0 and i != 0):
            send_result(range(max(start_process, i + start_process - _NO_PROCESSES_Per_EMAIL_), i + start_process))

    # Wait for all processes done
    for i in range(_NO_PARALLEL_PROCESSES_):
        processes[_NO_PROCESSES_ - _NO_PARALLEL_PROCESSES_ + i].join()
        progressBar.update(_NO_PROCESSES_ - _NO_PARALLEL_PROCESSES_ + i)
        save_last_process(_NO_PROCESSES_ - _NO_PARALLEL_PROCESSES_ + i)

    progressBar.finish()

    log('All processes done')
    send_log()
