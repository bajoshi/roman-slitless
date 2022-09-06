import numpy as np
import emcee
import corner

import matplotlib.pyplot as plt

from model_sn import model_sn


# This class came from stackoverflow
# SEE: https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python  # noqa
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


def read_pickle_make_plots_sn(object_type, ndim, args_obj, label_list,
                              truth_dict, savedir,
                              fitsmooth=False, orig_wav=None,
                              orig_spec=None, orig_ferr=None, 
                              plot_xlim_min=None, plot_xlim_max=None,
                              plot_ylim_min=None, plot_ylim_max=None):

    h5_path = savedir + 'emcee_sampler_' + object_type + '_resamp.h5'
    sampler = emcee.backends.HDFBackend(h5_path)

    samples = sampler.get_chain()
    print("\nRead in sampler:", h5_path)
    print("Samples shape:", samples.shape)

    # reader = emcee.backends.HDFBackend(pkl_path.replace('.pkl', '.h5'))
    # samples = reader.get_chain()
    # tau = reader.get_autocorr_time(tol=0)

    # Get autocorrelation time
    # Discard burn-in. You do not want to consider
    # the burn in the corner plots/estimation.
    tau = sampler.get_autocorr_time(tol=0)
    if not np.any(np.isnan(tau)):
        burn_in = int(2 * np.max(tau))
        thinning_steps = int(0.5 * np.min(tau))
    else:
        burn_in = 200
        thinning_steps = 30

    print("Average Tau:", np.mean(tau))
    print("Burn-in:", burn_in)
    print("Thinning steps:", thinning_steps)

    # construct truth arr and plot
    truth_arr = np.array([truth_dict['z'], truth_dict['phase'],
                          truth_dict['Av']])

    # plot trace
    fig1, axes1 = plt.subplots(ndim, figsize=(10, 6), sharex=True)

    for i in range(ndim):
        ax1 = axes1[i]
        ax1.plot(samples[:, :, i], "k", alpha=0.05)
        ax1.axhline(y=truth_arr[i], color='tab:red', lw=2.0)
        ax1.set_xlim(0, len(samples))
        ax1.set_ylabel(label_list[i], fontsize=15)
        ax1.yaxis.set_label_coords(-0.1, 0.5)

    axes1[-1].set_xlabel("Step number")

    fig1.savefig(savedir + 'emcee_trace_' + object_type + '.pdf', 
                 dpi=200, bbox_inches='tight')

    # Create flat samples
    flat_samples = sampler.get_chain(discard=burn_in,
                                     thin=thinning_steps, flat=True)
    print("\nFlat samples shape:", flat_samples.shape)

    # plot corner plot
    cq_z = corner.quantile(x=flat_samples[:, 0], q=[0.16, 0.5, 0.84])
    cq_day = corner.quantile(x=flat_samples[:, 1], q=[0.16, 0.5, 0.84])
    cq_av = corner.quantile(x=flat_samples[:, 2], q=[0.16, 0.5, 0.84])

    # print parameter estimates
    print(f"{bcolors.CYAN}")
    print("Parameter estimates:")
    print("Redshift: ", cq_z)
    print("Supernova phase [day]:", cq_day)
    print("Visual extinction [mag]:", cq_av)
    print(f"{bcolors.ENDC}")

    fig = corner.corner(flat_samples, quantiles=[0.16, 0.5, 0.84],
                        labels=label_list, 
                        label_kwargs={"fontsize": 14},
                        show_titles='True',
                        title_kwargs={"fontsize": 14},
                        truth_color='tab:red', truths=truth_arr,
                        smooth=0.5, smooth1d=0.5)

    # Extract the axes
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # Get the redshift axis
    # and edit how the errors are displayed
    ax_z = axes[0, 0]

    z_err_high = cq_z[2] - cq_z[1]
    z_err_low = cq_z[1] - cq_z[0]

    ax_z.set_title(r"$z \, =\,$" + r"${:.3f}$".format(cq_z[1]) +
                   r"$\substack{+$" + r"${:.3f}$".format(z_err_high) +
                   r"$\\ -$" + r"${:.3f}$".format(z_err_low) + r"$}$", 
                   fontsize=11)

    fig.savefig(savedir + 'corner_' + object_type + '.pdf', 
                dpi=200, bbox_inches='tight')

    # ------------ Plot 200 random models from the parameter 
    # space within +-1sigma of corner estimates
    # first pull out required stuff from args
    wav = args_obj[0]
    flam = args_obj[1]
    ferr = args_obj[2]

    fig3 = plt.figure(figsize=(9, 4))
    ax3 = fig3.add_subplot(111)

    ax3.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
    flam_label = r'$\mathrm{f_\lambda\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}]}$'
    ax3.set_ylabel(flam_label, fontsize=15)

    model_count = 0
    ind_list = []

    while model_count <= 200:

        ind = int(np.random.randint(len(flat_samples), size=1))
        ind_list.append(ind)

        # make sure sample has correct shape
        sample = flat_samples[ind]
        
        model_okay = 0

        sample = sample.reshape(3)

        # Get the parameters of the sample
        model_z = sample[0]
        model_day = sample[1]
        model_av = sample[2]

        # Check that the model is within +-1 sigma
        # of value inferred by corner contours
        if (model_z >= cq_z[0]) and (model_z <= cq_z[2]) and \
           (model_day >= cq_day[0]) and (model_day <= cq_day[2]) and \
           (model_av >= cq_av[0]) and (model_av <= cq_av[2]):

            model_okay = 1

        # Now plot if the model is okay
        if model_okay:

            m = model_sn(wav, sample[0], sample[1], sample[2])

            a = np.nansum(flam * m / ferr**2) / np.nansum(m**2 / ferr**2)
            m = m * a

            ax3.plot(wav, m, color='royalblue', lw=0.5, alpha=0.05, zorder=2)

            model_count += 1

    if fitsmooth:
        ax3.plot(orig_wav, orig_spec, color='k', lw=1.0, zorder=1)
        ax3.fill_between(orig_wav, orig_spec - orig_ferr,
                         orig_spec + orig_ferr,
                         color='gray', alpha=0.5, zorder=1)
        ax3.set_xlim(plot_xlim_min, plot_xlim_max)
        ax3.set_ylim(plot_ylim_min, plot_ylim_max)
    else:
        ax3.plot(wav, flam, color='k', lw=1.0, zorder=1)
        ax3.fill_between(wav, flam - ferr, flam + ferr, 
                         color='gray', alpha=0.5, zorder=1)

    # ADD LEGEND
    ax3.text(x=0.65, y=0.92, s='--- Simulated data', 
             verticalalignment='top', horizontalalignment='left', 
             transform=ax3.transAxes, color='k', size=12)
    ax3.text(x=0.65, y=0.85, s='--- 200 randomly chosen samples', 
             verticalalignment='top', horizontalalignment='left', 
             transform=ax3.transAxes, color='royalblue', size=12)

    fig3.savefig(savedir + 'emcee_overplot_' + object_type + '.pdf', 
                 dpi=200, bbox_inches='tight')

    # Close all figures
    fig1.clear()
    fig.clear()
    fig3.clear()

    plt.close(fig1)
    plt.close(fig)
    plt.close(fig3)

    return None
