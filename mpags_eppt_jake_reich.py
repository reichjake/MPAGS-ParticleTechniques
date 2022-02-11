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

def crystalBall(x, *p):
    alpha, n, mean, sigma = p

    A = ((n/abs(alpha))**n)*np.exp(-abs(alpha)**2/2)
    B = n/abs(alpha) - abs(alpha)
    C = (n/abs(alpha))*(1/(n-1))*np.exp(-abs(alpha)**2/2)
    D = np.sqrt(np.pi/2)*(1+erf(abs(alpha)/np.sqrt(2)))
    N = 1/(sigma*(C+D))

    f = np.piecewise(x,[(( x- mean)/sigma) > -alpha, (( x- mean)/sigma) <= -alpha] ,[N*np.exp(- (x-mean)**2/(2*sigma**2)), N*A*(B-((x-mean)/sigma))**(-n)])
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


    # zx_tracks_DC1 = {}
    # zx_tracks_DC2 = {}
    # n_multiple_hits = 0

    # for ev in range(0,len(DC1_z)): # loop through events
    #
    #     # store number of multiple hits in event
    #     DC1_N_multiple_hits = 0
    #     DC2_N_multiple_hits = 0
    #     DC1_multiple = {}



        # for z in range(0,len(DC1_z[ev])):
        #     n_multiple_hits = list(DC1_z[ev]).count(DC1_z[ev][z]) # number of multiple hits for DC1 chamber z
        #
        #     if(n_multiple_hits > 2): # duplicate hit in zth DC1
        #         print(DC1_z[ev])
        #         zipped_ev = zip(DC1_z[ev], DC1_x[ev])
        #         for L in range(0, len(DC1_z[ev])+1):
        #             for subset in itertools.combinations(zipped_ev[0], L):
        #                 if(len(subset) == len(DC1_z[ev])-(n_multiple_hits-1)):
        #                     print(subset)
        #         sys.exit()


            #if(DC1_z[ev][z] in np.delete(DC1_z[ev], z)): # check that this drift chamber has more than 1 hit
            #
            # n_multiple_hits = DC1_z[ev].count(DC1_z[ev][z]) # number of multiple hits for DC1 chamber z
            # if(n_multiple_hits > 1): # duplicate hit in zth DC1
            #     for z_multi in range(0,len(DC1_z[ev])):
            #         if(z_multi == z)

                #
                # DC1_multiple[str(DC1_z[ev][z])]



        #
        #     zx_tracks_DC1["eventNum_" + str(ev) ]["traj"] = [DC1_z[ev],DC1_x[ev]]
        # zx_tracks_DC2["eventNum_" + str(ev) ] = [DC2_z[ev],DC2_x[ev]]





    print("Number of events with more than one hit in a drift chamber = " + str(n_multiple_hits))

    return DC1_x, DC2_x


def plot_trajectories(zx_tracks, c,m, DC1 = True):


    z_range = np.arange(0,len(zx_tracks)*0.5,0.5)


    fig_zx = plt.figure()

    plt.scatter( z_range , zx_tracks, marker='x' )
    # define trajectory fit's x and z ranges
    z_fit = np.linspace(min(z_range),max(z_range), 1000)
    x_fit = np.linspace(min(zx_tracks),max(zx_tracks),1000)

    plt.plot(z_fit, c + m*z_fit, color = 'red')


    plt.xlabel("z [m]")
    plt.ylabel("x [m]")

    if(DC1 == True):
        plt.vlines(z_range, -100, 100)
        plt.savefig("DC1_zx"+suffix+".pdf")
    else:
        plt.vlines(z_range, -100, 100)
        plt.savefig("DC2_zx"+suffix+".pdf")


def plot_momentum_resolution(momenta, B, l):
    sigma_x = 100e-6 # x precision
    h = 5*0.5 # length of arm
    p_res =  (sigma_x/h)*(momenta/(0.3*B*l))


    fig = plt.figure(figsize = (8,6))

    data,bin_edges,patch  = plt.hist(p_res, bins = nbins)

    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

    p_res_range = np.linspace(min(p_res), max(p_res), 1000)
    popt, pcov = curve_fit(gauss, bin_centres, data, p0 = [1,1,1])

    plt.plot(p_res_range, gauss(p_res_range, *popt), color = 'red', label = 'Gaussian Fit')
    plt.legend(fontsize = f_size)
    plt.ylabel('Number of Events')

    textstr = '\n'.join((
    r'$A=%.3f \pm %.3f$' % (popt[0], np.sqrt(pcov[0][0]) ),
    r'$\mu=%.3f \pm %.3f$' % (popt[1],np.sqrt(pcov[1][1]) ),
    r'$\sigma=%.3f \pm %.3f$' % (abs(popt[2]),np.sqrt(pcov[2][2]) )))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    plt.text(max(bin_edges)*0.85, max(data)*0.8, textstr, fontsize=f_size,
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
    if("pb" in suffix):
        popt, pcov = curve_fit(gauss, bin_centres, data, p0 = [1,50,10])
        textstr = '\n'.join((
        r'$A=%.3f \pm %.3f$' % (popt[0], np.sqrt(pcov[0][0]) ),
        r'$\mu=%.3f \pm %.3f$' % (popt[1],np.sqrt(pcov[1][1]) ),
        r'$\sigma=%.3f \pm %.3f$' % (abs(popt[2]),np.sqrt(pcov[2][2]) )))
        plt.plot(momemta_range, gauss(momemta_range, *popt), color = 'red', label = 'Gaussian Fit')
    else:
        popt, pcov = curve_fit(crystalBall, bin_centres, data, p0 = [1,50,10,1])
        textstr = '\n'.join((
        r'$\alpha=%.3f \pm %.3f$' % (popt[0], np.sqrt(pcov[0][0]) ),
        r'$n=%.3f \pm %.3f$' % (popt[1],np.sqrt(pcov[1][1]) ),
        r'$mean=%.3f \pm %.3f$' % (popt[2],np.sqrt(pcov[2][2]) ),
        r'$\sigma=%.3f \pm %.3f$' % (abs(popt[3]),np.sqrt(pcov[3][3]) )))
        plt.plot(momemta_range, gauss(momemta_range, *popt), color = 'red', label = 'Crystal Ball Fit')
    plt.legend(fontsize = f_size)
    plt.xlabel(r'momentum $[GeV]$')
    plt.ylabel('Number of Events')



    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    plt.text(max(bin_edges)*0.7, max(data)*0.8, textstr, fontsize=f_size,
            verticalalignment='top', bbox=props)

    plt.savefig("p_dist"+ suffix +".pdf")

    return


if __name__ == "__main__":
    nbins = 35
    f_size = 13

    suffix = "_05T_100P_pbBeforeAndAfterMag"
    df = load_df(cwd, "/B5" + suffix)


    zx_tracks_DC1, zx_tracks_DC2 = zx_track_reco(df)
    momenta = []
    chi_square_DC1 = []
    chi_square_DC2 = []
    for ev in range(0,len(zx_tracks_DC1)):
        # extract hits from event
        hits_dc1 = zx_tracks_DC1[ev]
        hits_dc2 = zx_tracks_DC2[ev]

        # fit
        c_1, m_1, chisq_1 = fit_trajectory(np.arange(0,len(hits_dc1)*0.5,0.5), hits_dc1 )
        c_2, m_2, chisq_2 = fit_trajectory(np.arange(0,len(hits_dc2)*0.5,0.5), hits_dc2 )

        # plot_trajectories(hits_dc1, c_1,m_1, DC1 = True)
        # plot_trajectories(hits_dc2, c_2,m_2, DC1 = False)

        # get momentum
        momenta.append(calc_momentum(c_1, m_1, c_2, m_2, 0.5, 2))

        # store chisquare
        chi_square_DC1.append(chisq_1)
        chi_square_DC2.append(chisq_2)

    momenta = np.array(momenta)
    plot_momenta(momenta)
    plot_chisquare(chi_square_DC1,chi_square_DC2)
    # plot_momentum_resolution(momenta, 0.25 ,2)
