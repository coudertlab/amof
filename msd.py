"""
Module containing msd related methods
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 10:39:47 2019

@author: nicoc

prod v2
* used MSD algo with PBC
* put code into main func
"""

import ase
from ase import Atoms, data
from ase.io.trajectory import Trajectory

import matplotlib.pyplot as plt
import numpy as np


def main(traj_name):
    output_path = "../analysis/data/msd/" + traj_name
    
    traj = Trajectory('data/trajectory/'+traj_name+'.traj',mode='r')
    
    # Functions
    def position(atoms, atomic_number):
        """returns position arrays of a given species"""
        if atomic_number!=0:
            return atoms.get_positions()[atoms.get_atomic_numbers()==atomic_number]
        else:
            return atoms.get_positions()
    
   
    def MSD(trajectory, atomic_number = 0):
        """calculate MSD with real pos (stored in r) compared to PBC pos stored
        in ase (extracted with position)
        """
        r_0 = position(trajectory[0], atomic_number)
        r = np.zeros((len(trajectory), len(r_0), 3))
        r[0] = r_0 
        MSD = np.zeros(len(trajectory))
        for t in range(1, len(trajectory)):
            dr = np.zeros((len(r_0), 3))
            for j in range(3): #x,y,z
                a = trajectory[t].get_cell()[j,j]
                dr[:,j] = (position(trajectory[t], atomic_number) - r[t-1]%a)[:,j]
                for i in range(len(dr)):
                    if dr[i][j]>a/2:
                        dr[i][j] -= a
                    elif dr[i][j]<-a/2:
                        dr[i][j] += a
            r[t] = dr + r[t-1]
            MSD[t] = np.linalg.norm(r[t]-r_0)**2/len(r_0)
        return MSD
    
    Ag = data.atomic_numbers['Ag']
    Sb = data.atomic_numbers['Sb']
    Te = data.atomic_numbers['Te']
    
    
    
    t = np.linspace(0,4*len(traj)/1000,len(traj)) # in ps
    
    np.save(output_path + ".t", t)
    np.save(output_path + ".msd_X", MSD(traj))
    
    elements = [Ag,Sb,Te]
    for X in elements:
        X_str = data.chemical_symbols[X]
        MSD_array = MSD(traj, X)
        np.save(output_path + ".msd_" + X_str, MSD_array)
        plt.plot(t, MSD_array, label = X_str)
        
    plt.legend()
    plt.xlabel("Time (ps)")
    plt.ylabel("Mean-Square Displacement (${\AA}^{2}$)")
    
    
    MSD_array = MSD(traj)
    plt.plot(t, MSD_array, label = X_str)
    
    #plt.savefig("plots/MSD_"+traj_name+"_"+timestr+".png", dpi = 500)
    plt.show()


#traj_names = ["1.a1", "1.a2", "1.a3", "2.c1", "4.c"]
#for t in traj_names:
#    print(t, " started")
#    main(t)
main("4.2")