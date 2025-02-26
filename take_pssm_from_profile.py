import numpy as np

def get_PSSM(file_name):
    values = []
    with open(file_name) as f:
        lines = f.readlines()
        for i in range(3, 23):
            values.append(lines[i].strip().split('\t')[1:])

    return list(np.array(values).T)

