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


def circle(beta, x):
    return (x[0]-beta[0])**2 + (x[1]-beta[1])**2 -beta[2]**2

def calc_R(xc, yc, x,y):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def fit_trajectory(x,y):
    
    x_m = np.mean(x)
    y_m = np.mean(y)

    # initial guess for parameters
    R_m = calc_R(x_m, y_m,x,y).mean()
    beta0 = [ x_m, y_m, R_m]

    # for implicit function :
    #       data.x contains both coordinates of the points (data.x = [x, y])
    #       data.y is the dimensionality of the response
    lsc_data  = odr.Data(np.row_stack([x, y]), y=1)
    lsc_model = odr.Model(circle, implicit=True)
    lsc_odr   = odr.ODR(lsc_data, lsc_model, beta0)
    lsc_out   = lsc_odr.run()

    xc, yc, R = lsc_out.beta
    Ri = calc_R(xc, yc, x, y)
    residu = sum((Ri - R)**2)

    return xc, yc, R


def calc_momentum(B, R):
    e = 1.602176634e-19  # C
    p = e*B*R # kg.m.s^(-1)
    p_eV = (3e8)/(5.344286e-28)
    return p_eV


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



def xz_track_reco(df):
    # get x and z coordinates of hits in drift chambers
    DC1_x = df["Dc1HitsVector_x"].to_numpy()
    DC1_z = df["Dc1HitsVector_z"].to_numpy()

    DC2_x = df["Dc2HitsVector_x"].to_numpy()
    DC2_z = df["Dc2HitsVector_z"].to_numpy()



    xz_tracks_DC1 = {}
    xz_tracks_DC2 = {}
    n_backscatter_events = 0

    for ev in range(0,len(DC1_z)): # loop through events
        xz_tracks_DC1["eventNum_" + str(ev) ] = [] # store x coordinate of hits
        xz_tracks_DC2["eventNum_" + str(ev) ] = [] # store x coordinate of hits
        for nhits_DC1 in range(0,len(DC1_z[ev])):
            xz_tracks_DC1["eventNum_" + str(ev)].append(DC1_x[ev][nhits_DC1])
        for nhits_DC2 in range(0,len(DC2_z[ev])):
            xz_tracks_DC2["eventNum_" + str(ev)].append(DC2_x[ev][nhits_DC2])

    for ev in range(0,len(DC1_z)):
        if(
        len(np.unique(DC1_z[ev])) != len(DC1_z[ev])
        or len(np.unique(DC2_z[ev])) != len(DC2_z[ev])
        ):   # duplicates (backscatter)
            n_backscatter_events += 1
            del xz_tracks_DC1["eventNum_" + str(ev) ]
            del xz_tracks_DC2["eventNum_" + str(ev) ]



    print("Number of backscatter events = " + str(n_backscatter_events))

    return xz_tracks_DC1, xz_tracks_DC2


def plot_trajectories(xz_tracks, xc, yc, R, DC1 = True):

    fig_xz = plt.figure()
    for ev, hit in xz_tracks.items():
        hits = xz_tracks[ev]
        plt.plot( np.arange(0,len(hits)*0.5,0.5), hits, linewidth=0.2 )

        # define trajectory fit's x and z ranges
        z_fit = np.linspace(min(np.arange(0,len(hits)*0.5,0.5)),max(np.arange(0,len(hits)*0.5,0.5)), 1000)
        x_fit = np.linspace(min(hits),max(hits),1000)

        plt.plot(z_fit, circle([xc,yc,R],[z_fit, x_fit]), color = 'red')

    plt.xlabel("z [m]")
    plt.ylabel("x")

    if(DC1 == True):
        plt.ylim(-0.35, 0.5)
        plt.vlines(np.arange(0,len(hits)*0.5,0.5), -0.35, 0.5)
        plt.savefig("DC1_xz.pdf")
    else:
        plt.vlines(np.arange(0,len(hits)*0.5,0.5), -15, -6)
        plt.ylim(-15,-6)
        plt.savefig("DC2_xz.pdf")

if __name__ == "__main__":


    df = load_df(cwd, "/B5")
    xz_tracks_DC1, xz_tracks_DC2 = xz_track_reco(df)
    for ev in xz_tracks_DC1:
        hits = xz_tracks_DC1[ev]
        xc, yc, R = fit_trajectory(hits,np.arange(0,len(hits)*0.5,0.5) )
        print(xc,yc,R)
        print(calc_momentum(0.5, R))
        # plot_trajectories(xz_tracks_DC1, xc, yc, R, DC1 = True)
