import numpy as np
import pandas as pd
import math
import preprocessing as pp
import util
import train
import xgboost as xgb
import csv
import sys

pp.correlation('../data/train.csv')