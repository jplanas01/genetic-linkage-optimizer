from math import atan2, cos, sin, acos, asin, pi, floor, fsum
from genetic import GeneticAlgo
import random
from sys import stderr
import numpy as np

"""Conventions from:
    https://synthetica.eng.uci.edu/mechanicaldesign101/McCarthyNotes-2.pdf
"""

def valid_range(x):
    if x < -1 or x > 1:
        return False
    return True

def get_minmax(a, h, b, g):
    # Find out if linkage has an upper/lower limit for theta
    min_check = ((h-b) ** 2 - a**2 - g**2) / (2*a*g)
    max_check = ((h+b) ** 2 - a**2 - g**2) / (2*a*g)

    # If the checks are valid inputs for arccos (i.e., between -1 and 1 inclusive)
    # the linkage is constrained to certain values of theta.
    has_min = valid_range(min_check)
    has_max = valid_range(max_check)

    minim = None
    if has_min:
        minim = abs(acos(min_check))
        minim = (-minim, minim)
    maxim = None
    if has_max:
        maxim = abs(acos(max_check))
        maxim = (-maxim, maxim)

    # Set beginning and end of simulation depending on limits on theta
    # If no limits, run from 0 to 2pi
    if has_max and has_min:
        if minim[1] < maxim[1]:
            theta_min = minim[1]
            theta_max = maxim[1]
        else:
            theta_min = maxim[0]
            theta_max = minim[1]
    elif has_max:
        theta_min, theta_max = maxim
    elif has_min:
        theta_min = minim[0]
        theta_max = pi
    else:
        theta_min = 0
        theta_max = 2*pi

    return (theta_min, theta_max)


def sim_linkage(a, h, b, g):
    theta_min, theta_max = get_minmax(a, h, b, g)

    # Run through simulation of linkage, get position of point at intersection
    # of output crank and coupler.

    theta = np.arange(theta_min, theta_max, 0.01)
    npc = np.cos(theta)
    nps = np.sin(theta)
    A = 2*b*g - 2*a*b*npc
    B = 2*a*b*nps
    C = a**2 + b**2 + g**2 - h**2 - 2*a*g*npc
    check = -C / (A**2 + B**2) ** 0.5

    if np.any(np.logical_or(check < -1, check > 1)):
        #print('Non-real output angle psi.', file=stderr)
        return np.array([[0, 1e100]])

    delta = np.arctan2(B, A)
    psi1 = delta + np.arccos(check)
    #psi2 = delta - np.arccos(check)
    phi1 = np.arctan2(
            b * np.sin(psi1) - a * nps,
            g + b * np.cos(psi1) - a * npc)
    #phi2 = np.arctan2(
    #        b * np.sin(psi2) - a * np.sin(theta),
    #        g + b * np.cos(psi2) - a * np.cos(theta))
    x = a * npc + h * np.cos(phi1)
    y = a * nps + h * np.sin(phi1)

    return np.column_stack((x, y))
    
def vtarget(x):
    x = np.copy(x)
    targ1 = np.logical_or(x < -1, x > 1)
    targ2 = np.logical_not(targ1)

    x[targ1] = 1e99
    x[targ2] = 0.7

    return x

def simulate(child):
    sim_values = sim_linkage(*child)
    targets = vtarget(sim_values[:, 0])
    diffs = np.absolute(targets - sim_values[:, 1])

    if len(targets) < 100:
        return 1e100

    if np.max(sim_values[:, 0]) - np.min(sim_values[:, 0]) < 0.5:
        return 1e100

    return np.sum(diffs)

class LinkAlgo(GeneticAlgo):
    def gen_individual(self):
        child = []
        for i in range(4):
            child.append(random.random())
        return child

if __name__ == '__main__':
    random.seed(17)
    algo = LinkAlgo(10000, 20, 0.07, simulate)
    #algo.run()
    import cProfile
    cProfile.run('algo.run()', sort='cumtime')
    """
    best = sim_linkage(*algo.fitness[0][1])

    [print(x, y, sep=',') for x, y in best]
    print(algo.fitness[0][1], file=stderr)
    """
