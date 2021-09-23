import numpy as np


with open('exp_results.txt') as f:
    lines = f.readlines()
    cnt = 0
    exp = []
    for i in range(len(lines)):
        splitted = lines[i].split()
        if len(splitted) < 1:
            continue
        if '#' in splitted[0]:
            print(lines[i], end='')
            exp = []
            continue

        cnt += 1
        exp.append(float(splitted[0]))
        if cnt == 5:
            print(' '.join(splitted[1:-2]))
            print('min: {:.2f}, max: {:.2f}, mean: {:.2f}, std: {:.4f}\n'.format(min(exp), max(exp), sum(exp) / 5, np.std(exp)))
            cnt = 0
            exp = []

