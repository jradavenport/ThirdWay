
import numpy as np
import matplotlib.pyplot as plt
import emcee


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
