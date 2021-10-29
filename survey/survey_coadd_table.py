import numpy as np

# --------------------------------------
# This class came from stackoverflow
# SEE: https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
# --------------------------------------

total_visits = 9

print('\n   z-range       Mean z in range        DAY  : 0      5      10     15     20     25     30     35     40')
print('                                        VISIT: 0      1      2      3      4      5      6      7      8\n')

zmean_arr = np.arange(0.5, 3.5, 0.5)
starting_phase = -5

for zmean in zmean_arr:

    cosmic_time_dilation_rest_frame = 5 / (1 + zmean)
    phase_list = []

    zmin = '{:.2f}'.format(zmean - 0.25)
    zmax = '{:.2f}'.format(zmean + 0.25)

    print(zmin + ' <= z < ' + zmax, end='         ')
    print('{:.2f}'.format(zmean), end='                ')

    for i in range(total_visits):

        # Get current phase
        current_phase = starting_phase + cosmic_time_dilation_rest_frame * i

        # Formatting
        phase_to_print = '{:.2f}'.format(current_phase)
        if len(phase_to_print) == 4:
            phase_to_print = ' ' + phase_to_print

        # Printing
        if (current_phase >= -5.0) and (current_phase <= 5.0) and (i == total_visits - 1):
            print(f'{bcolors.CYAN}' + phase_to_print + f'{bcolors.ENDC}')
        elif (current_phase >= -5.0) and (current_phase <= 5.0):
            print(f'{bcolors.CYAN}' + phase_to_print + f'{bcolors.ENDC}', end='  ')
        elif (current_phase > 5.0) and (i == total_visits - 1):
            print(phase_to_print)
        elif (current_phase > 5.0) or (current_phase < -5.0):
            print(phase_to_print, end='  ')





