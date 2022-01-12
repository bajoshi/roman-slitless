import numpy as np


def get_template_inputs(template_name, verbose=False):

    # Read in the dummy template passed to pyLINEAR
    template_name_list = template_name.split('.txt')[0].split('_')

    # Get template properties
    if 'salt' in template_name:
        
        sn_av = float(template_name_list[-1].replace('p', '.').replace('av', ''))
        sn_z = float(template_name_list[-2].replace('p', '.').replace('z', ''))
        sn_day = int(template_name_list[-3].replace('day', ''))

        if verbose:
            print("Template file name:", template_name)
            print('SN info:')
            print('Redshift:', sn_z)
            print('Phase:', sn_day)
            print('Av:', sn_av)

        return [sn_z, sn_day, sn_av]

    else:

        galaxy_av = float(template_name_list[-1].replace('p', '.').replace('av',''))
        galaxy_met = float(template_name_list[-2].replace('p', '.').replace('met',''))
        galaxy_tau = float(template_name_list[-3].replace('p', '.').replace('tau',''))
        galaxy_age = float(template_name_list[-4].replace('p', '.').replace('age',''))
        galaxy_ms = float(template_name_list[-5].replace('p', '.').replace('ms',''))
        galaxy_z = float(template_name_list[-6].replace('p', '.').replace('z',''))

        galaxy_logtau = np.log10(galaxy_tau)

        if verbose:
            print("Template file name:", template_name)
            print('Galaxy info:')
            print('Redshift:', galaxy_z)
            print('Stellar mass:', galaxy_ms)
            print('Age:', galaxy_age)
            print('Tau:', galaxy_tau)
            print('Metallicity:', galaxy_met)
            print('Av:', galaxy_av)

        return [galaxy_z, galaxy_ms, galaxy_age, galaxy_logtau, galaxy_av]
