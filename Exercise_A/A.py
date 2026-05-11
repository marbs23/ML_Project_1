import random
import csv
import matplotlib.pyplot as plt
import os
import math

# DATA
def cargar_csv(ruta):
    data = []
    with open(ruta) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            row_to = [float(row[head]) for head in headers]
            data.append(row_to)
    return data, headers