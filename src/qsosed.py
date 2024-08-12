#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:56:36 2024

@author: Scott Hagen

Python version of the Xspec model QSOSED from Kubota & Done (2018).
Please see Kubota & Done, 2018, MNRAS, 480, 1247

(Note this inherits from agnsed)
"""

import numpy as np
import astropy.units as u

from agnsed import agnsed


class qsosed(agnsed):
    
    
    def __init__(self,
                 M=1e8,
                 dist=100,
                 log_mdot=-1,
                 a=0,
                 cos_inc=0.5,
                 fcol=1,
                 z=0):
        """
        
        Parameters
        ----------
        M : float
            Black hole mass - units : Msol
        dist : float
            Co-Moving Distance - units : Mpc
        log_mdot : float
            log mass accretion rate - units : Eddington
        a : float
            Dimensionless Black Hole spin - units : Dimensionless
        cos_inc : float
            cos inclination angle, as measured from the z-axis, with disc in the 
            x-y plane - units : Dimensionless
        fcol : float
            Colour temperature correction as described in Done et al. (2012)
            If -ve then follows equation 1 and 2 in Done et al. (2012).
            If +ve then assumes this to be constant correction over entire disc region
        z : float
            Redshift
        """

        #Read params
        self.M = M
        self.D, self.d = dist, (dist * u.Mpc).to(u.cm).value
        self.mdot = 10**(log_mdot)
        self.a = np.float64(a)
        self.inc = np.arccos(cos_inc)
        self.cosinc = cos_inc
        self.fcol = fcol
        self.z = z
        
        #Fixed pars
        self.kTe_h = 100 #keV
        self.kTe_w = 0.2 #keV
        self.gamma_w = 2.5

        #initiating constants
        self._set_constants()

        #Performing checks
        self._check_spin()
        self._check_inc()
        self._check_mdot()
        
        #Calculating disc params 
        self._calc_risco()
        self._calc_r_selfGravity()
        self._calc_Ledd()
        self._calc_efficiency()
        
        #physical conversion factors
        self.Mdot_edd = self.L_edd/(self.eta * self.c**2)
        self.Rg = (self.G * self.M)/(self.c**2)
        
        #Calculating disc regions
        self.dlog_r = 1/self.dr_dex
        self._set_rhot()
        self.r_w = 2*self.r_h
        self.r_out = self.r_sg
        self.hmax = min(100.0, self.r_h)
        
        if self.r_w > self.r_sg:
            self.r_w = self.r_sg
        
        
        #Energy/frequency grid
        self.Egrid = np.geomspace(self.Emin, self.Emax, self.numE)
        self.nu_grid = (self.Egrid * u.keV).to(u.Hz,
                                equivalencies=u.spectral()).value
        self.wave_grid = (self.Egrid * u.keV).to(u.AA,
                                equivalencies=u.spectral()).value
        
        self.E_obs = self.Egrid/(1+self.z)
        self.nu_obs = self.nu_grid/(1+self.z)
        self.wave_obs = self.wave_grid * (1+self.z)
        
        #Creating radal grid over disc and warm compton regions
        #using spacing of dr_dex
        self.dlog_r = 1/self.dr_dex
        self.logr_ad_bins = self._make_rbins(np.log10(self.r_w), np.log10(self.r_out))
        self.logr_wc_bins = self._make_rbins(np.log10(self.r_h), np.log10(self.r_w))
        self.logr_hc_bins = self._make_rbins(np.log10(self.risco), np.log10(self.r_h))
        
        
        #Setting X-ray power
        self._calc_Ldiss()
        self._calc_Lseed()
        self.rep = True
        self.Lx = self.Ldiss + self.Lseed
        
        self._set_gammah()
        
        
        #Energy/frequency grid
        self.Egrid = np.geomspace(self.Emin, self.Emax, self.numE)
        self.nu_grid = (self.Egrid * u.keV).to(u.Hz,
                                equivalencies=u.spectral()).value
        self.wave_grid = (self.Egrid * u.keV).to(u.AA,
                                equivalencies=u.spectral()).value
        
        self.E_obs = self.Egrid/(1+self.z)
        self.nu_obs = self.nu_grid/(1+self.z)
        self.wave_obs = self.wave_grid * (1+self.z)
        
        #Creating radal grid over disc and warm compton regions
        #using spacing of dr_dex
        self.dlog_r = 1/self.dr_dex
        self.logr_ad_bins = self._make_rbins(np.log10(self.r_w), np.log10(self.r_out))
        self.logr_wc_bins = self._make_rbins(np.log10(self.r_h), np.log10(self.r_w))
        self.logr_hc_bins = self._make_rbins(np.log10(self.risco), np.log10(self.r_h))
            
        
            
    def _check_mdot(self):
        """
        Checks is mdot within bounds.
        If log mdot < -1.65 then too small (The ENTIRE flow will be the corona..)
        If log mdot > 0.5 then the model is probably invalid (this is NOT a 
        super Eddington model...)
        """
        
        lmdot = np.log10(self.mdot)
        if lmdot >= -1.65 and lmdot <= 0.5:
            pass
        else:
            raise ValueError('mdot is out of bounds! \n'
                             'Require: -1.65 <= log mdot <= 0.5')      

    def _set_rhot(self):
        """
        Finds r_hot based on the condition L_diss = 0.02 Ledd, in the frame
        of the black hole
        Uses a refined radial grid, s.t we have more confidence in answer
        
        """

        dlr = 1/1000 #Somewhat refined grid for this calculation
        log_rall = self._make_rbins(np.log10(self.risco), np.log10(self.r_sg),
                                        dlog_r=dlr)
        Ldiss = 0.0
        i = 0
        while Ldiss < 0.02 * self.L_edd and i < len(log_rall) - 1:
            rmid = 10**(log_rall[i] + dlr/2)            
            dr = 10**(log_rall[i + 1]) - 10**(log_rall[i])
            
            Tnt4 = self.calc_Tnt(rmid)
            Ldiss += self.sigma_sb*Tnt4 * 4*np.pi*rmid*dr * self.Rg**2
            i += 1
            
        self.r_h = rmid
    
    
    def _set_gammah(self):
        """
        Sets the spectral index of the hot Compton region based off
        Beloborodov 1999 (see also Kubota & Done 2018)

        """
        
        self.gamma_h = (7/3) * (self.Ldiss/self.Lseed)**(-0.1)