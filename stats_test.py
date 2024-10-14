from stats_loader import StatsTracker, StatsLoader
import matplotlib.pyplot as plt

import os
import pandas as pd
from glob import glob
import numpy as np

directory_path = "D:\ŠOLA\Magisterj Modeli\\resnet18_realwaste_p_0.25"

json_files = glob(os.path.join(directory_path, '*.json'), recursive=True)

per2, acc2, epochs2 = np.zeros((18)),np.zeros((18)), np.zeros((18))
for file in json_files:
    try:
        stats_loader = StatsLoader()
        stats_tracker = stats_loader.load_from_file(file)

        for i, iteration in enumerate(stats_tracker.iterations):
            per2[i] = iteration.model_layers[-1].freezed_per
            acc2[i] += iteration.test_acc
            epochs2[i] += len(iteration.epochs)

    except Exception as e:
        print(f"Error reading {file}: {e}")

acc2 = acc2 / 5
epochs2 = epochs2 / 5

'''
for i, p in enumerate(per2):
    print(f"{i}: {p}")


for i, a in enumerate(acc2):
    print(f"{i}: {a}")

for i, e in enumerate(epochs2):
    print(f"{i}: {e}")
'''

directory_path = "D:\ŠOLA\Magisterj Modeli\\resnet18_realwaste_p_0.5"

json_files = glob(os.path.join(directory_path, '**', '*.json'), recursive=True)

per3, acc3, epochs3 = np.zeros((10)),np.zeros((10)), np.zeros((10))
for file in json_files:
    try:
        stats_loader = StatsLoader()
        stats_tracker = stats_loader.load_from_file(file)

        for i, iteration in enumerate(stats_tracker.iterations):
            per3[i] = iteration.model_layers[-1].freezed_per
            acc3[i] += iteration.test_acc
            epochs3[i] += len(iteration.epochs)

    except Exception as e:
        print(f"Error reading {file}: {e}")

acc3 = acc3 / 5
epochs3 = epochs3 / 5

directory_path = "D:\ŠOLA\Magisterj Modeli\\resnet18_realwaste_p_0.75"

json_files = glob(os.path.join(directory_path, '**', '*.json'), recursive=True)

per5, acc5, epochs5 = np.zeros((6)),np.zeros((6)), np.zeros((6))
for file in json_files:
    try:
        stats_loader = StatsLoader()
        stats_tracker = stats_loader.load_from_file(file)

        for i, iteration in enumerate(stats_tracker.iterations):
            per5[i] = iteration.model_layers[-1].freezed_per
            acc5[i] += iteration.test_acc
            epochs5[i] += len(iteration.epochs)

    except Exception as e:
        print(f"Error reading {file}: {e}")

acc5 = acc5 / 5
epochs5 = epochs5 / 5


#x = np.arange(1, 31)
#y1 = [0, 1, 4, 9, 16, 25]


plt.figure(figsize=(8, 6))

plt.plot(per2, acc2, label="0.25", marker='o')
plt.plot(per3, acc3, label="0.5", marker='s')
plt.plot(per5, acc5, label="0.75", marker='^')

plt.xlabel("Active Weights (%)")
plt.ylabel("Accuracy (%)")
plt.title("resnet18_realwaste")

plt.legend()

plt.gca().invert_xaxis()
plt.grid(True)
plt.show()