import numpy as np

def get_insertion_coords(num_to_insert):

    x_ins = np.zeros(num_to_insert)
    y_ins = np.zeros(num_to_insert)

    x_ins = np.random.randint(low=110, high=3985, size=num_to_insert)
    y_ins = np.random.randint(low=110, high=3985, size=num_to_insert)

    return x_ins, y_ins