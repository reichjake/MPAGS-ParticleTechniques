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
import itertools
import scipy




def line( z,c,m):
    return c + m*z



def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

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
        plt.savefig("DC1_zx.pdf")
    else:
        plt.vlines(z_range, -100, 100)
        plt.savefig("DC2_zx.pdf")


def plot_chisquare(chisquare_DC1, chisquare_DC2):
    fig = plt.figure()
    plt.hist(chisquare_DC1)
    plt.xlabel(r'$\chi^{2}$')
    plt.ylabel('Number of Events')
    plt.savefig("chi_square_DC1.pdf")

    fig = plt.figure()
    plt.hist(chisquare_DC2)
    plt.xlabel(r'$\chi^{2}$')
    plt.ylabel('Number of Events')
    plt.savefig("chi_square_DC2.pdf")
    return

def plot_momenta(momenta):

    fig = plt.figure()
    data,bin_edges,patch  = plt.hist(momenta, bins = 50)

    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

    momemta_range = np.linspace(min(momenta), max(momenta), 50)
    popt, pcov = curve_fit(gauss, bin_centres, data, p0 = [1,100,1])

    plt.plot(momemta_range, gauss(bin_centres, *popt), color = 'red', label = 'Gaussian Fit')
    # plt.xlim(0,1000)
    plt.xlabel(r'momentum $[GeV]$')
    plt.ylabel('Number of Events')
    plt.savefig("p_dist.pdf")

    return


if __name__ == "__main__":


    df = load_df(cwd, "/B5")
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


    plot_momenta(momenta)
    plot_chisquare(chi_square_DC1,chi_square_DC2)
