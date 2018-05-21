'''
Discrete variable MCMC sampler using Reversible Monte Carlo Markov Chain
'''
import logging

import numpy as np
from tqdm import tqdm


def MCMCMC(initial_array, loglike, args=None, kwargs=None, niter=1000, u0=0.5):
    '''
    Reversible jump MCMC algorithm

    This function calculates the reversible-jump MC3 process using following
    algorithm:

    1) Draw a uniform random variable `u`

    2) if ``u`` < 0.5 then take the *birth-death* step: random select a position
    in ``initial_array``, if it is 0 then turn it to 1 and turn it into 0 vice
    versa.

    3) if ``u`` > 0.5 then take the *swap* step, randomly select one location
    with 1 and exchange it to 0. If current state is full or null model, do not
    do any swap. Instead directly go to the next iteration

    4) calculate posterior from before and after, calculate the bayes factor
    between two models. Generate another uniform random variable `u2` and accept
    the proposal if `u2` < bayes factor.

    Parameters:
    -----------
    initial_array: list or numpy array
                   The initial starting array containing only 1 and 0.

    loglike: call-back function
             The call back function to calculate the log likelihood. The first
             parameter for this function needs to the array object like the
             ``initial_array``

    args: list
          Additional arguments for ``loglike`` function.

    kwargs: dictionary
            Additional arguments for ``loglike`` function.

    niter: int
           Total number of iterations for MCMC process.

    u0: float, valid range [0,1]
           The threshold to adjust birth/death and swap.
           random float < ``u0`` birth/death
           random float >= ``u0`` swap

    Return:
    -------
    results, acceptance_rates

    results: numpy array
             The array of the evolution trace of `initial_array`

    acceptance_rates: float
             The acceptance rate of MCMC jumps.
    '''
    logger = logging.getLogger('bms:MCMCMC')

    # Mopping args and kwargs if there is no additional argument
    if args is None:
        args = []

    if kwargs is None:
        kwargs = {}
    # Make sure we have correct data type into the function
    if not isinstance(args, list):
        raise TypeError('args needs to be a list of loglike function \
                        arguments, received type {0}'.format(type(args)))

    if not isinstance(kwargs, dict):
        raise TypeError('kargs needs to be a dict of loglike function \
                        arguments, received type {0}'.format(type(args)))

    if not isinstance(initial_array, np.ndarray):
        try:
            initial_array = np.array(initial_array)
        except:
            raise ValueError('Cannot convert initial_array \
                              to numpy array')
    total_iterations = niter
    old_array = initial_array
    results = []
    acceptance_count = 0
    cache = {}
    with tqdm(total=niter) as pbar:
        while niter > 0:
            proposal = old_array.copy()
            # Generate the first uniform variable u
            u = np.random.rand()
            if u <= u0:
                logger.debug('Birth/Death')
                logger.debug('Original array {0}'.format(proposal))
                pos = np.random.randint(0, len(proposal))
                if proposal[pos] == 1:
                    proposal[pos] = 0
                else:
                    proposal[pos] = 1
                logger.debug(
                    "after that, proposal becomes: {0}".format(proposal))
            else:
                logger.debug("Swap")
                if len(np.nonzero(proposal)[0]) == 0:
                    has_one = None
                else:
                    has_one = np.random.choice(np.nonzero(proposal)[0])
                if len(np.nonzero(1 - proposal)[0]) == 0:
                    has_zero = None
                else:
                    has_zero = np.random.choice(np.nonzero(1 - proposal)[0])
                while has_one == has_zero:
                    if len(np.nonzero(1 - proposal)[0]) == 0:
                        has_zero = np.random.choice(np.nonzero(proposal)[0])
                    else:
                        has_zero = np.random.choice(
                            np.nonzero(1 - proposal)[0])
                if has_one is None or has_zero is None:
                    logger.debug("Full model or null model. Stop Swap")
                    results.append(old_array)
                    niter -= 1
                    pbar.update(1)
                    continue
                else:
                    logger.debug('Original proposal: {0}'.format(proposal))
                    proposal[has_one] = 0
                    proposal[has_zero] = 1
                    logger.debug("after that, proposal becomes: \
                                {0}".format(proposal))

            # Calculate posterior probability
            if cache.get(tuple(old_array), None) is None:
                old_one_likelihood = loglike(old_array, *args, **kwargs)
                cache[tuple(old_array)] = old_one_likelihood
            else:
                old_one_likelihood = cache[tuple(old_array)]

            if cache.get(tuple(proposal), None) is None:
                proposal_likelihood = loglike(proposal, *args, **kwargs)
                cache[tuple(proposal)] = proposal_likelihood
            else:
                proposal_likelihood = cache[tuple(proposal)]
            bayes_factor = np.exp(proposal_likelihood - old_one_likelihood)
            logger.debug('Bayes Factor: {0}'.format(bayes_factor))
            acceptance_rate = np.min([1, bayes_factor])
            logger.debug('Acceptance rate: {0}'.format(acceptance_rate))

            # Jumping random variable
            jump = np.random.rand()
            logger.debug('random jump factor: {0}'.format(jump))
            if jump <= acceptance_rate:
                results.append(proposal)
                old_array = proposal
                acceptance_count += 1
                logger.debug('Accepted. Jump!')
            else:
                results.append(old_array)
                logger.debug('Not accepted. NO jump.')
            niter -= 1
            pbar.update(1)
    return results, acceptance_count / total_iterations
