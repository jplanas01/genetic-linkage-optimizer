from math import atan2, cos, sin, acos, asin, pi, floor, fsum
from genetic import GeneticAlgo
import random
from sys import stderr

"""Conventions from:
    https://synthetica.eng.uci.edu/mechanicaldesign101/McCarthyNotes-2.pdf
"""

def valid_range(x):
    if x < -1 or x > 1:
        return False
    return True

def calc_constants(theta, a, b, h, g):
    A = 2*b*g - 2*a*b*cos(theta)
    B = 2*a*b*sin(theta)
    C = a**2 + b**2 + g**2 - h**2 - 2*a*g*cos(theta)
    check = -C / (A**2 + B**2) ** 0.5

    return (A, B, C, check)

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
        theta_min = minim[1]
        theta_max = maxim[1]
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
    values = []
    for theta in range(floor(theta_min * 100), floor(theta_max * 100)):
        theta = theta / 100.0
        A, B, C, check = calc_constants(theta, a, b, h, g)
        if not valid_range(check):
            #print('Non-real output angle psi.', file=stderr)
            return [(0, 1e100)]

        delta = atan2(B, A)
        psi1 = delta + acos(check)
        psi2 = delta - acos(check)

        psi = psi1
        if psi1 < 0:
            psi = psi2

        cost = cos(theta)
        sint = sin(theta)

        phi1 = atan2(b * sin(psi) - a * sin(theta), 
                g + b * cos(psi) - a * cos(theta))
        phi2 = atan2(b * sin(psi2) - a * sin(theta), 
                g + b * cos(psi2) - a * cos(theta))

        phi = phi1
        if phi1 < 0:
            phi = phi2

        x = a * cos(theta) + h * cos(phi)
        y = a * sin(theta) + h * sin(phi)
        values.append((x, y))
    return values

def target(x):
    if x < -1 or x > 1:
        return 1e99
    return 0.7

def simulate(child):
    sim_values = sim_linkage(*child)
    sim_x = [x for x, _ in sim_values]
    sim_y = [y for _, y in sim_values]
    targets = [target(x) for x in sim_x]
    diffs = [abs(target - sim) for target, sim in zip(targets, sim_y)]

    if len(sim_x) > 0 and max(sim_x) - min(sim_x) < 0.5:
        return 1e100

    if len(diffs) > 100:
        return fsum(diffs)

    return 1e100

class LinkAlgo(GeneticAlgo):
    def gen_individual(self):
        child = []
        for i in range(4):
            child.append(random.random())
        return child

if __name__ == '__main__':
    algo = LinkAlgo(1000, 20, 0.07, simulate)
    import cProfile
    cProfile.run('algo.run()', sort='tottime')
    """
    best = sim_linkage(*algo.fitness[0][1])

    [print(x, y, sep=',') for x, y in best]
    print(algo.fitness[0][1], file=stderr)
    """
