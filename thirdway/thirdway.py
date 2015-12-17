
import numpy as np
import matplotlib.pyplot as plt
import emcee
import batman

def _gaus(x, a, b, x0, sigma):
    """
    Simple Gaussian function

    Parameters
    ----------
    x : float or 1-d numpy array
        The data to evaluate the Gaussian over
    a : float
        the amplitude
    b : float
        the constant offset
    x0 : float
        the center of the Gaussian
    sigma : float
        the width of the Gaussian

    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b


def RemoveTransit(data):
  '''
  Use M-A model parameters, subtract from light curve
  '''
  
  return flatdata


def FitBumps(data, nspots=4):
  '''
  Use emcee to fit data with N spots in given transit
  '''
  
  return bestfitmodel


def Datum2Lon(t):
  '''
  Given parameters of system:
  Prot, Porb, ephem, etc
  Convert a given time (or array of times) *in transit* to star surface coordinates (lon, lat)
  '''
  
  return (lon, lat)
