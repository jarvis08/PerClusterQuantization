import numpy as np


with open('exp_results.txt') as f:
    lines = f.readlines()
    cnt = 0
    exp = []
    to_average = ''
    for i in range(len(lines)):
        splitted = lines[i].split()
        if len(splitted) < 1:
            continue
        if '##' in splitted[0]:
            continue
        elif '#' == splitted[0]:
            print(lines[i], end='')
            exp = []
            continue

        cur_params = splitted[2:splitted.index('Best-epoch:')]
        if cur_params == to_average:
            cnt += 1
            exp.append(float(splitted[0]))
            if cnt == 5:
                print(' '.join(splitted[1:-2]))
                print('min: {:.2f}, max: {:.2f}, mean: {:.2f}, std: {:.4f}\n'.format(min(exp), max(exp), sum(exp) / 5, np.std(exp)))
                cnt = 0
                exp = []
        else:
            to_average = cur_params
            cnt = 1
            exp = []
            exp.append(float(splitted[0]))

