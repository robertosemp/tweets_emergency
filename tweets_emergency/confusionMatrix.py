import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class confusionMatrix():
    def __init__(self, threshold):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.threshold = threshold
        self.accuracy = 0 
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.total = 0

    def update(self, real, prediction):

        if prediction.shape != real.shape:
            prediction = torch.flatten(prediction)
        prediction = prediction > self.threshold
        real = real > self.threshold

        self.TP += sum(real * prediction).item()
        self.TN += sum(~real * ~prediction).item()
        self.FP += sum(~real * prediction).item()
        self.FN += sum(real * ~prediction).item()
        
        
    def calc_metrics(self):
        self.total = self.TP + self.TN + self.FP + self.FN 
        self.precision = self.TP / (self.TP + self.FP)
        self.recall = self.TP / (self.TP + self.FN)
        self.accuracy = (self.TP + self.TN) / (self.total)
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        
        
    def output(self):
        print("-----------------------------")
        print("true positives: " + str(self.TP / self.total))
        print("true negatives: " + str(self.TN / self.total))
        print("false positives: " + str(self.FP / self.total))
        print("false negatives: " + str(self.FN / self.total))
        print("-----------------------------")
        print("accuracy: " + str(self.accuracy))
        print("precision: " + str(self.precision))
        print("recall: " + str(self.recall))
        print("f1: " + str(self.f1))
        print("-----------------------------")