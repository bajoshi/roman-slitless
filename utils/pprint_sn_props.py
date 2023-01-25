def print_sn_props(cat, idx):
    print("--" * 22)
    print('   #   SegID   Y106-mag     z     1hr-SNR')
    print("--" * 22)
    for i in range(len(idx)):
        print('{:4d}'.format(i+1), '  ',
              '{:4d}'.format(cat['SNSegID'][idx][i]), '  ',
              '{:.3f}'.format(cat['Y106mag'][idx][i]), '  ',
              '{:.3f}'.format(cat['z_true'][idx][i]), '  ',
              '{:>.2f}'.format(cat['SNR1200'][idx][i]))
    print("--" * 22)