# Titanic Passenger Survival Model
# Copyright (C) 2013 Appasaurus

import csv
import numpy as np
import scipy as sp

from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

def readFile(filename):
    f = csv.reader(open(filename, 'rb'))
    header = f.next()

    data = []
    for row in f:
        data.append(row[1:])
    
    return np.array(data)

def parseRow(row):
    # Y 0: class
    # N 1: name
    # Y 2: sex
    # Y 3: age (int or nil)
    # Y 4: SibSp
    # Y 5: Parch
    # N 6: Ticket
    # Y 7: Fare (float)
    # N 8: Cabin (multiple, single, nil)
    # Y 9: Embarked (C, S, Q, nil)
    
    # Cabin class
    cabin_class = row[0]

    # Sex
    sex = 0 if row[2] == 'female' else 2

    # Age
    age = 1000 if row[3] == '' else row[3]

    # Number of siblings/spouse on board
    siblings_spouse = 0 if row[4] == '' else row[4]

    # Number of parents/children on board
    parents_children = 0 if row[5] == '' else row[5]

    # Fare
    fare = 0 if row[7] == '' else row[7]

    # Point of embarcation
    embarcation = 0
    if row[9] == 'S':
        embarcation = 1
    elif row[9] == 'Q':
        embarcation = 2
    elif row[9] == 'C':
        embarcation = 3

    return [cabin_class, sex, age, siblings_spouse, parents_children, fare, embarcation]


def constructDataset(data):
    dataset = ClassificationDataSet(7)
    for row in data:
        dataset.addSample(parseRow(row[1:]), row[0])

    return dataset
    
def main():
    training_data = readFile('data/train.csv')
    dataset = constructDataset(training_data)
    network = buildNetwork(7, 10, 1)
    trainer = BackpropTrainer(network, dataset)
    trainer_error = trainer.train() #trainUntilConvergence()

    print 'Trainer Error:', trainer_error

    predicted_survival = 0
    predicted_dead = 0
    
    actual_survival = 0
    actual_dead = 0
    
    correct = 0
    false_positives = 0
    false_negatives = 0
    
    num_cases = len(training_data)

    for i in range(num_cases):
        predicted = network.activate(parseRow(training_data[i][1:]))[0]
        actual = int(training_data[i][0])
    
        if predicted > 0.5:
            predicted = 1
        else:
            predicted = 0

        if predicted == 1 and actual == 0:
            false_positives += 1
        if predicted == 0 and actual == 1:
            false_negatives += 1
        if predicted == actual:
            correct += 1

        if predicted == 0:
            predicted_dead += 1
        if predicted == 1:
            predicted_survival += 1

        if actual == 0:
            actual_dead += 1
        if actual == 1:
            actual_survival += 1

    print 'Predicted Survival:', predicted_survival, '/', num_cases
    print 'Actual Survival:', actual_survival, '/', num_cases
    print ''
    print 'Correct Cases:', correct, '/', num_cases, '=', float(correct)/num_cases * 100, '%'
    print 'False Positives:', false_positives, '/', num_cases
    print 'False Negatives:' , false_negatives, '/', num_cases

if __name__ == '__main__':
    main()
