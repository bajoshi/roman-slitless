import pymc3 as pm

import arviz as az
import numpy as np
import matplotlib.pyplot as plt

import time
import os
import sys

# Required for the more complicated test
import theano
import theano.tensor as tt
import emcee
import corner


# define your super-complicated model that uses loads of external codes
def my_model(theta, x):
    """
    A straight line!

    Note:
        This function could simply be:

            m, c = thetha
            return m*x + x

        but I've made it more complicated for demonstration purposes
    """
    m, c = theta  # unpack line gradient and y-intercept

    line = m * x + c

    return line


# define your really-complicated likelihood function that uses loads of external codes
def my_loglike(theta, x, data, sigma):
    """
    A Gaussian log-likelihood function for a model with parameters given in theta
    """

    model = my_model(theta, x)

    return -(0.5/sigma**2)*np.sum((data - model)**2)


# define a theano Op for our likelihood function
class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, x, sigma):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.x, self.data, self.sigma)

        outputs[0][0] = np.array(logl)  # output the log-likelihood


def basic_test():

    # Initialize model
    with pm.Model() as model:

        # E.g., to define a flat prior
        # with some limits
        #z = pm.Uniform('z', lower=0.0, upper=3.0)

        # prior
        mu = pm.Normal('mu', mu=0, sigma=1)
        
        # Observed data
        obs = pm.Normal('obs', mu=mu, sigma=1, observed=np.random.randn(1000))

        # Run sampler
        idata = pm.sample(2000, tune=1500, return_inferencedata=True)

    print(idata.posterior.dims)

    az.plot_trace(idata)

    summary = az.summary(idata)

    print("Summary:")
    print(summary)

    plt.show()

    return None


if __name__ == '__main__':

    print("\n##################")
    print("WARNING: only use pymc3 in the base conda env. NOT in astroconda.")
    print("##################\n")
    time.sleep(2)

    print(f"Running on PyMC3 v{pm.__version__}")
    print(f"Running on ArviZ v{az.__version__}")
    print("\n")

    # -------------------
    #basic_test()

    # ------------------- More complicated test
    print("Doing the blackbox example from PyMC3 docs.")
    print("See: https://github.com/pymc-devs/pymc-examples/blob/main/examples/case_studies/blackbox_external_likelihood.ipynb")
    
    ##############################
    # set up our data
    N = 10000  # number of data points
    sigma = 1.0  # standard deviation of noise
    x = np.linspace(0.0, 9.0, N)
        
    mtrue = 0.4  # true gradient
    ctrue = 3.0  # true y-intercept
        
    truemodel = my_model([mtrue, ctrue], x)
        
    # make data
    np.random.seed(716742)  # set random seed, so the data is reproducible each time
    data = sigma * np.random.randn(N) + truemodel
        
    ndraws = 5000  # number of draws from the distribution
    nburn = 1000  # number of "burn-in points" (which we'll discard)

    nchains = 4
    ncores = 4

    ndim = 2

    label_list = ['m', 'c']
    
    # create our Op
    logl = LogLike(my_loglike, data, x, sigma)

    # use PyMC3 to sampler from log-likelihood
    with pm.Model() as model:
        # uniform priors on m and c
        m = pm.Uniform("m", lower=-10.0, upper=10.0)
        c = pm.Uniform("c", lower=-10.0, upper=10.0)
    
        # convert m and c to a tensor vector
        theta = tt.as_tensor_variable([m, c])
    
        # use a DensityDist (use a lamdba function to "call" the Op)
        #pm.DensityDist("likelihood", my_logl, observed={"v": theta})
        like = pm.Potential("like", logl(theta))
    
    with model:
        trace = pm.sample(ndraws, cores=ncores, chains=nchains, tune=nburn, discard_tuned_samples=True)
        print(pm.summary(trace).to_string())
    
        # put the chains in an array (for later!)
        samples_pymc3 = np.vstack((trace["m"], trace["c"])).T

        # plot traces
        m_trace = trace["m"].reshape((nchains, ndraws))
        c_trace = trace["c"].reshape((nchains, ndraws))

        samples = np.array([m_trace.T, c_trace.T])

        samples = samples.reshape((ndraws, nchains, ndim))
        print(samples.shape)

        fig1, axes1 = plt.subplots(ndim, figsize=(10, 6), sharex=True)

        for i in range(ndim):
            ax1 = axes1[i]
            ax1.plot(samples[:, :, i], "k", alpha=0.05)
            ax1.set_xlim(0, len(samples))
            ax1.set_ylabel(label_list[i])
            ax1.yaxis.set_label_coords(-0.1, 0.5)

        axes1[-1].set_xlabel("Step number")

        # corner plot 
        fig = corner.corner(samples_pymc3, labels=[r"$m$", r"$c$"],
            truths=[mtrue, ctrue])

        plt.show()

    sys.exit(0)