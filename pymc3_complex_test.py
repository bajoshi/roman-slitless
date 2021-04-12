import numpy as np
import pymc3 as pm
import theano.tensor as tt
#from theano.tests import unittest_tools as utt

import arviz as az
import matplotlib.pyplot as plt
import corner

class Loglike(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, data):
        self.data = data
        self.loglike_grad = LoglikeGrad(self.data)

    def perform(self, node, inputs, outputs):
        theta, = inputs
        mu, sigma = theta
        logp = -len(self.data)*np.log(np.sqrt(2.0*np.pi)*sigma)
        logp += -np.sum((self.data - mu)**2.0)/(2.0*sigma**2.0)
        outputs[0][0] = np.array(logp)

    def grad(self, inputs, grad_outputs):
        theta, = inputs
        grads = self.loglike_grad(theta)
        return [grad_outputs[0]*grads]

class LoglikeGrad(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, data):
        self.data = data

    def perform(self, node, inputs, outputs):
        theta, = inputs
        mu, sigma = theta
        dmu = np.sum(self.data - mu)/sigma**2.0
        dsigma = np.sum((self.data - mu)**2.0)/sigma**3.0 - len(self.data)/sigma
        outputs[0][0] = np.array([dmu, dsigma])

def main():
    mu_true = 5.0
    sigma_true = 1.0
    data = np.random.normal(loc=mu_true, scale=sigma_true, size=10000)
    loglike = Loglike(data)
    #utt.verify_grad(loglike, [np.array([3.0, 2.0])])
    # verify_grad passes with no errors
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=4.0, sigma=2.0, testval=4.0)
        sigma = pm.HalfNormal('sigma', sigma=5.0, testval=2.0)
        theta = tt.as_tensor_variable([mu, sigma])
        like = pm.Potential('like', loglike(theta))
    with model:
        trace = pm.sample()
        print(pm.summary(trace).to_string())

        # plot the traces
        _ = az.plot_trace(trace, lines={"mu": mu_true, "sigma": sigma_true})

        # put the chains in an array (for later!)
        samples = np.vstack((trace["mu"], trace["sigma"])).T

        # corner plot 
        fig = corner.corner(samples, labels=[r"$mu$", r"$\sigma$"],
            truths=[mu_true, sigma_true])

        plt.show()

if __name__ == "__main__":
    print("pymc3 version: ", pm.__version__)
    main()