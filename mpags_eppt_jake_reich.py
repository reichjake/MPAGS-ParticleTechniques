from __future__ import print_function
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd+'/ML_Python/')
import numpy as np
import argparse

import ROOT
from ROOT.Math import PxPyPzEVector
from ROOT import TFile, TH1F, TH2F

import uproot3
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import  odr
from scipy.optimize import curve_fit
from scipy import signal
from scipy.special import erf
import itertools
import scipy




def line( z,c,m):
    return c + m*z



def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def skewNorm(x, *p):
    A, alpha, omega, xi = p

    arg = (x - xi)/omega
    phi = A*np.exp(-arg**2/2)
    Phi = (1/2)*(1+erf(alpha*arg/np.sqrt(2)))

    return (2/omega)*phi*Phi

def crystalBall(x, *p):
    alpha, n, mean, sigma = p

    x = np.array(x).astype(np.float)
    alpha = -1*alpha

    A = ((n/abs(alpha))**n)*np.exp(-1*abs(alpha)**2/2)
    B = n/abs(alpha) - abs(alpha)
    C = (n/abs(alpha))*(1/(n-1))*np.exp(-1*abs(alpha)**2/2)
    D = np.sqrt(np.pi/2)*(1+erf(abs(alpha)/np.sqrt(2)))
    N = 1/(sigma*(C+D))

    f = np.piecewise(x,[(( x- mean)/sigma) > -1*alpha, (( x- mean)/sigma) <= -1*alpha] ,
    [lambda x: N*np.exp(- (x-mean)**2/(2*sigma**2)), lambda x: N*A*(B-((x-mean)/sigma))**(-n)])
    # if( (( x- mean)/sigma) > -alpha ):
    #     f = N*np.exp(- (x-mean)**2/(2*sigma**2))
    # else:
    #     f = N*A*(B-((x-mean)/sigma))**(-n)

    return f

def fit_trajectory(z,x):

    # # fit
    popt, pcov = curve_fit(line, z ,x , p0 = [1,1])
    # get chi squared
    chi_squared = np.sum((np.polyval([popt[0],  popt[1]], z) - x) ** 2)

    return popt[0],  popt[1], chi_squared


def calc_deflection_angle(m_1,m_2):
    return abs(np.arctan((m_1 + m_2)/(1+(m_1*m_2))))

def calc_momentum(c_1, m_1, c_2, m_2, B, l):

    theta = calc_deflection_angle(m_1,m_2)
    p = (0.3*B*l)/(2*np.sin(theta/2))

    return p


def load_df(path, filename):
    print("Reading in file: " + filename + "..." )


    try:
        file = uproot3.open(path+filename + ".root")["B5"]
    except FileNotFoundError:
        print("Wrong file or file path")
        return -1
    else:
        try:
            keys = file.keys()
            keys_decode = [key.decode('ASCII') for key in keys]
            dataframe = file.pandas.df(keys, flatten=False)
            dataframe.columns = keys_decode
        except KeyError:
            print("Key Error")
            return -1

    print("done...")

    return dataframe



def zx_track_reco(df):


    # get x and z coordinates of hits in drift chambers
    DC1_x = df["Dc1HitsVector_x"].to_numpy()/1000
    DC1_z = df["Dc1HitsVector_z"].to_numpy()
    DC2_x = df["Dc2HitsVector_x"].to_numpy()/1000
    DC2_z = df["Dc2HitsVector_z"].to_numpy()

    n_multiple_hits = 0
    remove_events = []
    for ev in range(0,len(DC1_z)):
        if(
        len(np.unique(DC1_z[ev])) != len(DC1_z[ev])
        or len(np.unique(DC2_z[ev])) != len(DC2_z[ev])
        ):   # duplicates

            n_multiple_hits += 1
            remove_events.append(ev)

    DC1_z = np.delete(DC1_z, remove_events)
    DC2_z = np.delete(DC2_z, remove_events)
    DC1_x = np.delete(DC1_x, remove_events)
    DC2_x = np.delete(DC2_x, remove_events)

    print("Number of events with more than one hit in a drift chamber = " + str(n_multiple_hits))

    return DC1_x, DC2_x


def plot_trajectories(zx_tracks,  DC1 = True):




    df = load_df(cwd, "/B5" + suffix)
    zx_tracks_DC1, zx_tracks_DC2 = zx_track_reco(df)

    if(DC1 == True):
        zx_tracks = zx_tracks_DC1
    else:
        zx_tracks = zx_tracks_DC2

    fig_zx = plt.figure()


    for ev in range(0,len(zx_tracks)):
        z_range = np.arange(0,len(zx_tracks[ev])*0.5,0.5)
        plt.plot( z_range , zx_tracks[ev])
        # define trajectory fit's x and z ranges
        # z_fit = np.linspace(min(z_range),max(z_range), 1000)
        # x_fit = np.linspace(min(zx_tracks),max(zx_tracks),1000)
        #
        # plt.plot(z_fit, c + m*z_fit, color = 'red')


    plt.xlabel("z [m]")
    plt.ylabel("x [m]")

    if(DC1 == True):
        # plt.vlines(z_range, -100, 100)
        plt.savefig("DC1_zx"+suffix+".pdf")
    else:
        # plt.vlines(z_range, -100, 100)
        plt.savefig("DC2_zx"+suffix+".pdf")


def plot_momentum_resolution(momenta,m_1_arr,m_2_arr, B, l):
    sigma_x = 100e-6 # x precision
    h = 1 + (np.array(m_1_arr)*np.array(m_2_arr)) # denominator of tan(theta)
    p_res =  (sigma_x/h)*(np.array(momenta)/(0.3*B*l))


    fig = plt.figure(figsize = (8,6))

    data,bin_edges,patch  = plt.hist(p_res, bins = nbins)

    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

    p_res_range = np.linspace(min(p_res), max(p_res), 1000)
    popt, pcov = curve_fit(gauss, bin_centres, data, p0 = [160,0.01,0.001])

    plt.plot(p_res_range, gauss(p_res_range, *popt), color = 'red', label = 'Gaussian Fit')
    plt.legend(fontsize = f_size)
    plt.ylabel('Number of Events')

    textstr = '\n'.join((
    r'$A=%.5f \pm %.5f$' % (popt[0], np.sqrt(pcov[0][0]) ),
    r'$\mu=%.5f \pm %.5f$' % (popt[1],np.sqrt(pcov[1][1]) ),
    r'$\sigma=%.5f \pm %.5f$' % (abs(popt[2]),np.sqrt(pcov[2][2]) )))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    plt.text(max(bin_edges)*0.91, max(data)*0.8, textstr, fontsize=f_size,
            verticalalignment='top', bbox=props)


    plt.xlabel(r'$\frac{\sigma_{p}}{p}$', fontsize = 15)
    plt.ylabel('Number of Events')
    plt.savefig("p_resolution"+suffix+".pdf")

def plot_chisquare(chisquare_DC1, chisquare_DC2):
    fig = plt.figure()
    plt.hist(chisquare_DC1)
    plt.xlabel(r'$\chi^{2}$')
    plt.ylabel('Number of Events')
    plt.savefig("chi_square_DC1"+suffix+".pdf")

    fig = plt.figure()
    plt.hist(chisquare_DC2)
    plt.xlabel(r'$\chi^{2}$')
    plt.ylabel('Number of Events')
    plt.savefig("chi_square_DC2" + suffix+".pdf")
    return

def plot_momenta(momenta):

    fig = plt.figure()
    data,bin_edges,patch  = plt.hist(momenta, bins = nbins)

    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

    momemta_range = np.linspace(min(momenta), max(momenta), 1000)
    if("pb" not in suffix):
        popt, pcov = curve_fit(gauss, bin_centres, data, p0 = [1,200,10])
        textstr = '\n'.join((
        r'$A=%.3f \pm %.3f$' % (popt[0], np.sqrt(pcov[0][0]) ),
        r'$\mu=%.3f \pm %.3f$' % (popt[1],np.sqrt(pcov[1][1]) ),
        r'$\sigma=%.3f \pm %.3f$' % (abs(popt[2]),np.sqrt(pcov[2][2]) )))
        plt.plot(momemta_range, gauss(momemta_range, *popt), color = 'red', label = 'Gaussian Fit')
    else:
        popt, pcov = curve_fit(skewNorm, bin_centres, data, p0 = [800.0, 3.0, 20.0, 80.0])
        textstr = '\n'.join((
        r'$A=%.3f \pm %.3f$' % (popt[0], np.sqrt(pcov[0][0]) ),
        r'$\alpha=%.3f \pm %.3f$' % (popt[1], np.sqrt(pcov[1][1]) ),
        r'$\omega=%.3f \pm %.3f$' % (popt[2], np.sqrt(pcov[2][2]) ),
        r'$\xi=%.3f \pm %.3f$' % (popt[3], np.sqrt(pcov[3][3]) )))
        plt.plot(momemta_range, skewNorm(momemta_range, *popt), color = 'red', label = 'Skew Normal Fit')
        plt.xlim(0,250)
    plt.legend(fontsize = f_size)
    plt.xlabel(r'momentum $[GeV]$')
    plt.ylabel('Number of Events')



    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    plt.text(max(bin_edges)*0.35, max(data)*0.8, textstr, fontsize=f_size,
            verticalalignment='top', bbox=props)

    plt.savefig("p_dist"+ suffix +".pdf")

    return

def getCalorimeterEnergy(df):

    ECalEnergy = df["ECEnergy"].to_numpy()/1000
    HCalEnergy = df["HCEnergy"].to_numpy()/1000
    ECalEnergy_vec = df["ECEnergyVector"].to_numpy()/1000
    HCalEnergy_vec = df["HCEnergyVector"].to_numpy()/1000

    return ECalEnergy,HCalEnergy,ECalEnergy_vec,HCalEnergy_vec


def plotCalDists(ECal, HCal, ECal_vec, HCal_vec):
    range = np.linspace(0, 1, nbins)
    fig = plt.figure()
    plt.hist(ECal, label = "ECal",bins =  range, histtype = "step", color = "blue")
    #plt.hist(HCal, label = "HCal", bins =  range, histtype = "step", color = "orange")
    plt.legend()
    # plt.xlim(0,0.6)
    plt.xlabel(r'Total Energy Deposited [GeV]')
    plt.ylabel('Number of Events')
    plt.savefig("calorimeter_"+ suffix +"_ecal_.pdf")


if __name__ == "__main__":
    nbins = 50
    f_size = 13

    suffix = "_05T_100P"
    df = load_df(cwd, "/B5" + suffix)

    # zx_tracks_DC1, zx_tracks_DC2 = zx_track_reco(df)
    # momenta = []
    # chi_square_DC1 = []
    # chi_square_DC2 = []
    # m_1_arr = []
    # m_2_arr = []
    # for ev in range(0,len(zx_tracks_DC1)):
    #     # extract hits from event
    #     hits_dc1 = zx_tracks_DC1[ev]
    #     hits_dc2 = zx_tracks_DC2[ev]
    #
    #     # fit
    #     c_1, m_1, chisq_1 = fit_trajectory(np.arange(0,len(hits_dc1)*0.5,0.5), hits_dc1 )
    #     c_2, m_2, chisq_2 = fit_trajectory(np.arange(0,len(hits_dc2)*0.5,0.5), hits_dc2 )
    #
    #     m_1_arr.append(m_1)
    #     m_2_arr.append(m_2)
    #
    #
    #
    #
    #     # get momentum
    #     momenta.append(calc_momentum(c_1, m_1, c_2, m_2, 0.5, 2))
    #
    #     # store chisquare
    #     chi_square_DC1.append(chisq_1)
    #     chi_square_DC2.append(chisq_2)
    #
    # plot_trajectories(hits_dc1, DC1 = True)
    # plot_trajectories(hits_dc2, DC1 = False)
    #
    # momenta = np.array(momenta)
    # plot_momenta(momenta)
    # # plot_chisquare(chi_square_DC1,chi_square_DC2)
    # plot_momentum_resolution(momenta,m_1_arr,m_2_arr, 0.5 ,2)



    ECalEnergy,HCalEnergy,ECalEnergy_vec,HCalEnergy_vec = getCalorimeterEnergy(df)
    plotCalDists(ECalEnergy, HCalEnergy, ECalEnergy_vec,HCalEnergy_vec)
    # for ev in range(0,len(ECalEnergy)):
