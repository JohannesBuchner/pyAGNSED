#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:45:24 2024

@author: Scott Hagen


Python version of the Xspec model AGNSED from Kubota & Done (2018).
Please see Kubota & Done, 2018, MNRAS, 480, 1247
"""


import numpy as np
import warnings
import astropy.constants as const
import astropy.units as u

from myNTHCOMP import mdonthcomp


class agnsed:
    
    Emin = 1e-4
    Emax = 1e4
    numE = 2000
    
    A = 0.3 #disc albedo - fixed at 0.3
    
    dr_dex = 50 #grid spacing - N points per decade
    
    default_units = 'cgs'
    units = 'cgs'
    
    #Stop all runtime warnings (Happens when calculating bbody and numbers get 
    #too small! - doesn't affect output)
    warnings.filterwarnings('ignore')
    
    def __init__(self,
                 M=1e8,
                 dist=100,
                 log_mdot=-1,
                 a=0,
                 cos_inc=0.5,
                 kTe_hot=100,
                 kTe_warm=0.2,
                 gamma_hot=1.7,
                 gamma_warm=2.7,
                 r_hot=10,
                 r_warm=20,
                 log_rout=-1,
                 fcol=1,
                 h_max=10,
                 rep=True,
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
            cos inclination angle
        kTe_hot : float
            Electron temp for hot Compton region - units : keV
        kTe_warm : float
            Electron temp for warm Compton region - units : keV
        gamma_hot : float
            Spectral index for hot Compton region
        gamma_warm : float
            Spectral index for warm Compton region
        r_hot : float
            Outer radius of hot Compton region - units : Rg
        r_warm : float
            Outer radius of warm Compton region - units : Rg
        log_rout : float
            log of outer disc radius - units : Rg
        fcol : float
            Colour temperature correction as described in Done et al. (2012)
            If -ve then follows equation 1 and 2 in Done et al. (2012).
            If +ve then assumes this to be constant correction over entire disc region
        h_max : float
            Scale height of hot Compton region - units : Rg
        rep : bool
            Switch for including X-ray re-processing off the disc
        z : float
            Redshift

        """
        
        #Read Pars
        self.M = M
        self.D, self.d = dist, (dist*u.Mpc).to(u.cm).value
        self.mdot = 10**(log_mdot)
        self.a = np.float64(a)
        self.inc = np.arccos(cos_inc)
        self.kTe_h = kTe_hot
        self.kTe_w = kTe_warm
        self.gamma_h = gamma_hot
        self.gamma_w = gamma_warm
        self.r_h = r_hot
        self.r_w = r_warm
        self.r_out = 10**(log_rout)
        self.fcol = fcol
        self.hmax = h_max
        self.rep = rep
        self.z = z
        
        self.cosinc = cos_inc
        
        #initiating constants (in correct units)
        self._set_constants()
        
        #Performing checks
        self._check_spin()
        self._check_inc()
        
        
        #Calculating Disc pars
        self._calc_risco()
        self._calc_r_selfGravity()
        self._calc_Ledd()
        self._calc_efficiency()
        
        if log_rout < 0:
            self.r_out = self.r_sg #setting to self gravity if log_rout < 0
        
        if r_warm == -1:
            self.r_w = self.risco
        
        if r_hot == -1:
            self.r_h = self.risco
            self.hmax = 0
            
        self._check_rw()
        self._check_risco()
        self._check_hmax()
        
        
        #physical conversion factors
        self.Mdot_edd = self.L_edd/(self.eta * self.c**2)
        self.Rg = (self.G * self.M)/(self.c**2)
        
        
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
        
        #Calculating X-ray power
        self._calc_Ldiss()
        self._calc_Lseed()
        self.Lx = self.Ldiss + self.Lseed
        
        
        
    def _set_constants(self):
        """
        Sets the physical constants in cgs units as object attributes 
        """
        
        self.G = (const.G * const.M_sun).to(u.cm**3 / u.s**2).value #cm^3/Msol/s^2
        self.sigma_sb = const.sigma_sb.to(u.erg/u.s/u.K**4/u.cm**2).value 
        self.c = const.c.to(u.cm/u.s).value #speed of light, cm/s
        self.h = const.h.to(u.erg*u.s).value #Plank constant, erg s
        self.k_B = const.k_B.to(u.erg/u.K).value #Boltzmann constant, #erg/K
        self.m_e = const.m_e.to(u.g).value  #electron mass, g
        
    
    
    ###########################################################################
    #---- Performing checks on certain parameters. 
    #     To ensure that we are within both physical limits 
    #     (i.e -0.998 <= a <= 0.998)
    ###########################################################################
        
    def _check_spin(self):
        if self.a >= -0.998 and self.a <= 0.998:
            pass
        else:
            raise ValueError('Spin ' + str(self.a) + ' not physical! \n'
                             'Must be within: -0.998 <= a_star <= 0.998')
        
        
    def _check_inc(self):
        if self.cosinc <= 1 and self.cosinc >= 0.09:
            pass
        else:
            raise ValueError('Inclination out of bounds - will not work with kyconv! \n'
                             'Require: 0.09 <= cos(inc) <= 0.98 \n'
                             'Translates to: 11.5 <= inc <= 85 deg')
        
        
    def _check_rw(self):
        if self.r_w >= self.r_h:
            pass
        else:
            print('WARNING r_warm < r_hot ---- Setting r_warm = r_hot')
            self.r_w = self.r_h
            
        if self.r_w <= self.r_out:
            pass
        else:
            print('WARNING r_warm > r_out ----- Setting r_warm = r_out')
            self.r_w = self.r_out
        
        
    def _check_risco(self):
        if self.r_h >= self.risco:
            pass
        else:
            print('WARNING r_hot < r_isco ----- Setting r_hot = r_isco')
            self.r_h = self.risco
            self.hmax = 0
            
        if self.r_w >= self.risco:
            pass
        else:
            print('WARNING! r_warm < r_isco ----- Settin r_warm = r_isco')
            self.r_w = self.risco
        
        
    def _check_hmax(self):
        if self.hmax <= self.r_h:
            pass
        else:
            print('WARNING! hmax > r_h ------- Setting hmax = r_h')
            self.hmax = self.r_h
    
    
    
    ###########################################################################
    #---- Unit Handling
    #     Note, only affects output. Internal calculations still done in cgs
    ###########################################################################
    
    def set_units(self, new_unit='cgs'):
        """
        Re-sets default units. ONLY affects attributes extracted through the
        getter methods
        
        Note, the only difference between setting cgs vs counts is in spectra

        Parameters
        ----------
        new_unit : {'cgs','cgs_wave', 'SI', 'counts'}, optional
            The default unit to use. The default is 'cgs'.
             * cgs: the luminosity spectral density is in `u.erg/u.s/u.Hz`, use `agn.nu_grid * u.Hz`
             * counts: the luminosity spectral density is in `u.keV/u.s/u.keV`, use `agn.Egrid * u.keV`
             * cgs_wave: the luminosity spectral density is in `u.erg/u.s/u.AA`, use `agn.wavegrid * u.AA`
             * SI: the luminosity spectral density is in `u.W/u.s/u.Hz`, use `agn.nu_grid * u.Hz`
            Integrated luminosities correspondingly lose /Hz or /AA.
            Fluxes correspondingly are per luminosity distance squared.

        """
        #Checking valid units
        unit_lst = ['cgs', 'cgs_wave', 'SI', 'counts']
        if new_unit not in unit_lst:
            print('Invalid Unit!!!')
            print(f'Valid options are: {unit_lst}')
            print('Setting as default: cgs')
            new_unit = 'cgs'
            
        self.units = new_unit
    
    
    def _to_newUnit(self, L, as_spec=True):
        """
        Sets input luminosity/spectrum to new output units

        Parameters
        ----------
        L : float OR array
            Input lum/spectrum.

        Returns
        -------
        Lnew : float OR array
            In new units.
        unit : str
            new unit (in case the currently set unit is not desired)
        as_spec : bool
            If True, then input should be erg/s/Hz
            If false, then input should be erg/s

        """
        #If spectral density
        if as_spec:
            if self.units == 'cgs':
                Lnew = L
                
            elif self.units == 'counts':
                Lnew = (L*u.erg/u.s/u.Hz).to(u.keV/u.s/u.keV,
                                             equivalencies=u.spectral()).value
                
                if np.ndim(L) == 1:
                    Lnew /= self.Egrid
                else:
                    Lnew /= self.Egrid[:, np.newaxis]
            
            elif self.units == 'cgs_wave':
                Lnew = (L*u.erg/u.s/u.Hz).to(u.erg/u.s/u.AA,
                                equivalencies=u.spectral_density(self.nu_grid*u.Hz)).value
            
            else:
                Lnew = L*1e-7
        
        #If just a luminosity
        else:
            if self.units == 'cgs' or self.units == 'counts':
                Lnew = L
            else:
                Lnew = L*1e-7
        
        return Lnew
    
    
    def _to_flux(self, L):
        """
        Converts to a flux - takes redshift into accounts    

        Parameters
        ----------
        L : float OR array
            Luminosity to be converted.

        Returns
        -------
        f : float OR array
            Flux seen by observer

        """
        
        if self.units == 'cgs' or self.units == 'counts':
            d = self.d #distance in cm
        else:
            d = self.d/100 #distance in m
        
        f = L/(4*np.pi*d**2)# * (1+self.z))
        return f
        
    
    ###########################################################################
    #---- Disc Properties
    ###########################################################################
    
    def _calc_Ledd(self):
        """
        Calculate eddington Luminosity

        """
        Ledd = 1.39e38 * self.M #erg/s
        self.L_edd = Ledd
    
    
    def _calc_risco(self):
        """
        Calculating innermost stable circular orbit for a spinning
        black hole. Follows Page and Thorne (1974). Note, can also be reffered
        to as r_ms, for marginally stable orbit
        
        return r_isco as property - so will be called in __init__

        """
        Z1 = 1 + (1 - self.a**2)**(1/3) * (
            (1 + self.a)**(1/3) + (1 - self.a)**(1/3))
        Z2 = np.sqrt(3 * self.a**2 + Z1**2)

        self.risco = 3 + Z2 - np.sign(self.a) * np.sqrt(
            (3 - Z1) * (3 + Z1 + 2*Z2))


    def _calc_r_selfGravity(self):
        """
        Calcultes the self gravity radius according to Laor & Netzer 1989
        
        NOTE: Assuming that \alpha=0.1 

        """
        alpha = 0.1 #assuming turbulence NOT comparable to sound speed
        #See Laor & Netzer 1989 for more details on constraining this parameter
        m9 = self.M/1e9
        self.r_sg = 2150 * m9**(-2/9) * self.mdot**(4/9) * alpha**(2/9)    
        
    
    def _calc_efficiency(self):
        """
        Calculates the accretion efficiency eta, s.t L_bol = eta Mdot c^2
        Using the GR case, where eta = 1 - sqrt(1 - 2/(3 r_isco)) 
            Taken from: The Physcis and Evolution of Active Galactic Nuceli,
            H. Netzer, 2013, p.38
        """
        
        self.eta = 1 - np.sqrt(1 - 2/(3*self.risco))
        
    
    def _calc_NTparams(self, r):
        """
        Calculates the Novikov-Thorne relativistic factors.
        see Active Galactic Nuclei, J. H. Krolik, p.151-154
        and Page & Thorne (1974)

        """
        y = np.sqrt(r)
        y_isc = np.sqrt(self.risco)
        y1 = 2 * np.cos((1/3) * np.arccos(self.a) - (np.pi/3))
        y2 = 2 * np.cos((1/3) * np.arccos(self.a) + (np.pi/3))
        y3 = -2 * np.cos((1/3) * np.arccos(self.a))

        
        B = 1 - (3/r) + ((2 * self.a)/(r**(3/2)))
        
        C1 = 1 - (y_isc/y) - ((3 * self.a)/(2 * y)) * np.log(y/y_isc)
        
        C2 = ((3 * (y1 - self.a)**2)/(y*y1 * (y1 - y2) * (y1 - y3))) * np.log(
            (y - y1)/(y_isc - y1))
        C2 += ((3 * (y2 - self.a)**2)/(y*y2 * (y2 - y1) * (y2 - y3))) * np.log(
            (y - y2)/(y_isc - y2))
        C2 += ((3 * (y3 - self.a)**2)/(y*y3 * (y3 - y1) * (y3 - y2))) * np.log(
            (y - y3)/(y_isc - y3))
        
        C = C1 - C2
        
        return C/B
    
    
    def calc_Tnt(self, r):
        """
        Calculates Novikov-Thorne disc temperature^4 at radius r. 
        
        Parameters
        ----------
        r : float OR array
            Radius at which to evaluate the temperature - Units : Rg
        fmd : float, optional
            Factor by which to modulate the mass accretion rate
            i.e fmd = mdot(t)/mdot_intrinsic
            If 1, then T4 will be the intrinisic NT temp
        
        Returns
        -------
        T4 : float OR array
            Novikov-Thorne temperature of the system, at radius r
            Note, this is at T^4 - Units : K^4
            
        """

        Rt = self._calc_NTparams(r)
        const_fac = (3 * self.G * self.M * self.mdot * self.Mdot_edd)/(
            8 * np.pi * self.sigma_sb * (r * self.Rg)**3)
        
        T4 = const_fac * Rt
        return T4
    
    
    def calc_Trep(self, r, Lx):
        """
        Calculates the re-processed temperature at r for a given X-ray
        luminosity

        Parameters
        ----------
        r : float OR array
            Radial coordinate - units : Rg
        Lx : float OR array
            X-ray luminosity seen at point r, phi at time t - units : erg/s
        Returns
        -------
        T4rep : float OR array
            Reprocessed temperature at r, to the power 4

        """
        
        R = r * self.Rg
        H = self.hmax * self.Rg
        
        Frep = (Lx)/(4*np.pi * (R**2 + H**2))
        Frep *= H/np.sqrt(R**2 + H**2)
        Frep *= (1 - self.A) 
        
        T4rep = Frep/self.sigma_sb

        return T4rep * (1 - self.A)
    
    
    def calc_fcol(self, Tm):
        """
        Calculates colour temperature correction following Eqn. (1) and 
        Eqn. (2) in Done et al. (2012)

        Parameters
        ----------
        Tm : float
            Max temperature at annulus (ie T(r)) - units : K.

        Returns
        -------
        fcol_d : float
            colour temperature correction at temperature T

        """
        if Tm > 1e5:
            #For this region follows Eqn. (1)
            Tm_j = self.k_B * Tm
            Tm_keV = (Tm_j * u.erg).to(u.keV).value #convert to units consitant with equation
            
            fcol_d = (72/Tm_keV)**(1/9)
        
        elif Tm < 1e5 and Tm > 3e4:
            #For this region follows Eqn. (2)
            fcol_d = (Tm/(3e4))**(0.82)
        
        else:
            fcol_d = 1
        
        return fcol_d
    
    
    ###########################################################################
    #---- Handling binning
    ###########################################################################
    
    def _make_rbins(self, logr_in, logr_out, dlog_r=None):
        """
        Creates an array of radial bin edges, with spacing defined by dr_dex
        Calculates the bin edges from r_out and down to r_in. IF the bin
        between r_in and r_in+dr is less than dlog_r defined by dr_dex, then
        we simply create a slightly wider bin at this point to accomodate
        for the difference

        Parameters
        ----------
        logr_in : float
            Inner radius of model section - units : Rg.
        logr_out : float
            Outer radius of model section - units : Rg.

        Returns
        -------
        logr_bins : 1D-array
            Radial bin edges for section - units : Rg.

        """
        if dlog_r is not None:
            dlr = dlog_r
        else:
            dlr = self.dlog_r
        
        
        i = logr_out
        logr_bins = np.array([np.float64(logr_out)]) 
        while i > logr_in:
            r_next_edge = i - dlr
            logr_bins = np.insert(logr_bins, 0, r_next_edge)
            i = r_next_edge

       
        if logr_bins[0] != logr_in:
            if logr_bins[0] < logr_in:
                if len(logr_bins) > 1:
                    logr_bins = np.delete(logr_bins, 0)
                    logr_bins[0] = logr_in
                else:
                    logr_bins[0] = logr_in
            else:
                logr_bins[0] = logr_in
        
        return logr_bins
    
    
    def new_ear(self, new_es):
        """
        Defines new energy grid if necessary

        Parameters
        ----------
        ear : 1D-array
            New energy grid - units : keV.

        """
        
        Ebins = new_es
        dEs = self.Ebins[1:] - self.Ebins[:-1]
        
        self.Egrid = Ebins[:-1] + 0.5 * dEs
        self.nu_grid = (self.Egrid * u.keV).to(u.Hz,
                                equivalencies=u.spectral()).value
        self.nu_obs = self.nu_grid/(1 + self.z) #Observers frame
        self.E_obs = self.Egrid/(1 + self.z)
        
        self.Emin = min(self.Egrid)
        self.Emax = max(self.Egrid)
        self.numE = len(self.Egrid)
    
            
        
    def set_radialResolution(self, Ndex):
        """
        Re-sets the radial binning within the code
        
        This is a de-bugging aide!!!!

        Parameters
        ----------
        Ndex : float or int
            Number of bins per radial decade

        Returns
        -------
        None.

        """
        
        self.dr_dex = Ndex
        self.__init__(self.M, self.D, np.log10(self.mdot), self.a,
                      self.cosinc, self.kTe_h, self.kTe_w, self.gamma_h, 
                      self.gamma_w, self.r_h, self.r_w, np.log10(self.r_out), 
                      self.fcol, self.hmax, self.rep, self.z)
        
    
    ###########################################################################
    #---- Emission from standard disc region
    ###########################################################################
    
    def _bb_radiance(self, T):
        """
        Calculates the thermal emission for a given temperature T

        Parameters
        ----------
        T : float OR array
            Temperature - Units : K

        Returns
        -------
        pi * Bnu : 2D-array, shape=(len(nus), len(Ts))
            Black-body emission spectrum - Units : erg/s/cm^2/Hz

        """
        
        pre_fac = (2*self.h*self.nu_grid[:, np.newaxis]**3)/(self.c**2)
        exp_fac = np.exp((self.h*self.nu_grid[:, np.newaxis])/(self.k_B*T)) - 1
        Bnu = pre_fac / exp_fac
        
        return np.pi * Bnu
    
    
    def _do_disc_annuli(self, r, dr):
        """
        Calculates the emission from an annulus within the standard disc region

        If array input gives the emission from a set of annuli

        Parameters
        ----------
        r : float OR array
            Geometric midpoint of radial bin
            Units : Rg
        dr : float OR array
            Radial bin width (linear)
            Units : Rg

        Returns
        -------
        Lnu_ann : 2D-array, shape=(len(Egrid), len(r))
            Luminosity spectral density
            Units : ergs/s/Hz

        """
        
        T4_ann = self.calc_Tnt(r)
        if self.rep:
            T4_ann = T4_ann + self.calc_Trep(r, self.Lx)
        
        Tann = T4_ann**(1/4)
        if self.fcol < 0:
            fcol_r = self.calc_fcol(Tann)
        else:
            fcol_r = self.fcol
        
        Tann *= fcol_r
        bb_ann = self._bb_radiance(Tann)/(fcol_r**4)
        Lnu_ann = 4*np.pi*r*dr * bb_ann * self.Rg**2 * (self.cosinc/0.5)
        
        return Lnu_ann
    

    def _do_discSpec(self):
        """
        Calculates the total contribution to the SED from the standard disc
        region

        Returns
        -------
        Lnu_dsc : array
            Luminosity spectral density
            Units : ergs/s/Hz
            
        """
        
        dr_bins = 10**(self.logr_ad_bins[1:]) - 10**(self.logr_ad_bins[:-1]) #linear width
        rmids = 10**(self.logr_ad_bins[:-1] + self.dlog_r/2) #gemotric bin center
        
        if len(rmids) == 0:
            self.Lnu_disc = np.zeros(len(self.Egrid))
        else:
            Lnu_arr = self._do_disc_annuli(rmids, dr_bins)
            self.Lnu_disc = np.sum(Lnu_arr, axis=-1)
        
        return self.Lnu_disc
    
    
    ###########################################################################
    #---- Emission from warm Compton region
    ###########################################################################
    
    def _do_warm_annuli(self, r, dr):
        """
        Calculates the emission from an annulus in the warm Comptonisation
        region

        Parameters
        ----------
        r : float OR array
            Geometric midpoint of radial bin
            Units : Rg
        dr : float OR array
            Radial bin width (linear)
            Units : Rg

        Returns
        -------
        Lnu_ann : 2D-array, shape=(len(Egrid), len(r))
            Luminosity spectral density
            Units : ergs/s/Hz
            
        """
        
        T4_ann = self.calc_Tnt(r)
        if self.rep:
            T4_ann = T4_ann + self.calc_Trep(r, self.Lx)
        
        kTann = self.k_B * T4_ann**(1/4) #ergs
        kTann = (kTann*u.erg).to(u.keV).value #keV
        
        if isinstance(kTann, np.ndarray):
            pass
        else:
            kTann = np.array([kTann])
            
        #transforming gamma and kTe_w to correct format for nthcomp
        gamma_arr = np.full(len(kTann), self.gamma_w)
        kTe_arr = np.full(len(kTann), self.kTe_w)

        #calculating compton spectrum
        ph_nth = mdonthcomp(self.Egrid, gamma_arr, kTe_arr, kTann)
        ph_nth = (ph_nth * u.erg/u.s/u.keV).to(u.erg/u.s/u.Hz,
                                               equivalencies=u.spectral()).value
        
        #Getting physical normalisation
        norm = self.sigma_sb * T4_ann * 4*np.pi*r*dr * self.Rg**2 * (self.cosinc/0.5)
        radience = np.trapz(ph_nth, self.nu_grid, axis=0)
        
        Lnu_ann = norm * (ph_nth/radience)
        return Lnu_ann
    
    
    def _do_warmSpec(self):
        """
        Calculates the total spectrum from the warm Compton region

        Returns
        -------
        Lnu_wrm : array
            Luminosity spectral density
            Units : ergs/s/Hz

        """
        
        dr_bins = 10**(self.logr_wc_bins[1:]) - 10**(self.logr_wc_bins[:-1]) #linear width
        rmids = 10**(self.logr_wc_bins[:-1] + self.dlog_r/2) #gemotric bin center
        
        if len(rmids) == 0:
            #no warm region
            self.Lnu_warm = np.zeros(len(self.Egrid))
        else:
            Lnu_arr = self._do_warm_annuli(rmids, dr_bins)
            self.Lnu_warm = np.sum(Lnu_arr, axis=-1)
        
        return self.Lnu_warm
    
    
    ###########################################################################
    #---- Emission from Hot Compton region
    ###########################################################################
    
    def _calc_Ldiss(self):
        """
        Calculates the total power dissipated within the corona
        Stores as attribute for later use
        
        (in units erg/s)

        """
        
        dr_bins = 10**(self.logr_hc_bins[1:]) - 10**(self.logr_hc_bins[:-1]) #linear width
        rmids = 10**(self.logr_hc_bins[:-1] + self.dlog_r/2) #gemotric bin center
        
        T4s = self.calc_Tnt(rmids)
        self.Ldiss = np.sum(self.sigma_sb*T4s * 4*np.pi*rmids*dr_bins*self.Rg**2)


    def _calc_Lseed(self):
        """
        Calculates the seed photon luminosity seen by the hot corona
        Stores as attribute for later use (in ergs/s)

        """
        
        logr_tot_bins = self._make_rbins(np.log10(self.r_h), np.log10(self.r_out))
        hc = min(self.r_h, self.hmax)
        
        drs = 10**(logr_tot_bins[1:]) - 10**(logr_tot_bins[:-1]) #linear width
        rmids = 10**(logr_tot_bins[:-1] + self.dlog_r/2) #gemotric bin center
        
        #calculating covering fraction of each annulus seen by corona
        #see Appendix in Hagen & Done 2023b, MNRAS, 525, 3455
        cov_frac = np.zeros(len(rmids))
        theta0 = np.zeros(len(rmids))
        
        theta0[rmids>=hc] = np.arcsin(hc/rmids)
        cov_frac[rmids>=hc] = theta0[rmids>=hc] - 0.5*np.sin(2*theta0[rmids>=hc])
        
        #getting annular contribution
        T4_ann = self.calc_Tnt(rmids)
        Fr = self.sigma_sb * T4_ann
        Lr = 4*np.pi*rmids*drs * Fr * (cov_frac/np.pi) * self.Rg**2
        
        self.Lseed = np.sum(Lr)


    def kTseed_hot(self):
        """
        Calculates seed photon temperature for the Hot Compton region

        Returns
        -------
        kT_seed : float
            Seed photon temperature for hot Compton
            Units : keV

        """
        
        T4_edge = self.calc_Tnt(self.r_h)
        Tedge = T4_edge**(1/4)
        
        kT_edge = self.k_B * Tedge #ergs
        kT_edge = (kT_edge*u.erg).to(u.keV).value
        
        if self.r_w != self.r_h:
            #If warm Compton exists - include Compton y-param
            ysb = (self.gamma_w * (4/9))**(-4.5)
            kT_seed = np.exp(ysb) * kT_edge
        
        else:
            kT_seed = kT_edge
            if self.fcol < 0:
                fcol_r = self.calc_fcol(Tedge)
            else:
                fcol_r = self.fcol
            
            kT_seed *= fcol_r
        
        return kT_seed

    
    def _do_hotSpec(self):
        """
        Calculates hot Compton spectrum

        Returns
        -------
        Lnu_hot : array
            Luminosity spectral density
            Units : ergs/s/Hz

        """
        
        kT_seed = self.kTseed_hot()
        if kT_seed == 0:
            self.Lnu_hot = np.zeros(len(self.Egrid))
        else:
            ph_nth = mdonthcomp(self.Egrid, np.array([self.gamma_h]), 
                            np.array([self.kTe_h]), np.array([kT_seed]))[:, 0]
            ph_nth = (ph_nth * u.erg/u.s/u.keV).to(u.erg/u.s/u.Hz, 
                                            equivalencies=u.spectral()).value
        
            radience = np.trapz(ph_nth, self.nu_grid)
            self.Lnu_hot = self.Lx * (ph_nth/radience)
        
        return self.Lnu_hot
        
    
    ###########################################################################
    #---- Methods for getting SEDs and SED components
    #     Takes into account unit choices
    ###########################################################################
    
    
    def get_SED(self, as_flux=False):
        """
        Extracts SED in currently set units (see set_units() method)

        Parameters
        ----------
        as_flux : Bool, optional
            Whether to convert output to flux units (includes redshift)
            The default is False.

        Returns
        -------
        fLnu : array
            Flux/Luminosity density of SED

        """
        
        fLnu_dsc = self.get_SEDcomponent('disc', as_flux=as_flux)
        fLnu_wrm = self.get_SEDcomponent('warm', as_flux=as_flux)
        fLnu_hot = self.get_SEDcomponent('hot', as_flux=as_flux)
        
        fLnu = fLnu_dsc + fLnu_wrm + fLnu_hot
        
        return fLnu
    
    
    def get_SEDcomponent(self, component, as_flux=False):
        """
        Extracts SED component from single emission region.
        i.e Either standard disc (disc), warm Compton (warm), or
        hot Compton (hot)
        
        Uses whatever units are currently set (see set_units() method)

        Parameters
        ----------
        component : {'disc', 'warm', 'hot'}
            Which component to extract
        as_flux : Bool, optional
            Whether to convert output to flux units (includes redshift)
            The default is False

        Returns
        -------
        fLnu_cmp : array
            Flux/Luminosity density of SED component

        """
        
        if component in ['disc', 'warm', 'hot']:
            pass
        else:
            raise ValueError('component must be: disc, warm, or hot')
        
        
        if hasattr(self, f'Lnu_{component}'):
            Lnu_cmp = getattr(self, f'Lnu_{component}')
        else:
            Lnu_cmp = getattr(self, f'_do_{component}Spec')()
        
        fLnu_cmp = self._to_newUnit(Lnu_cmp, as_spec=True)
        if as_flux == True:
            fLnu_cmp = self._to_flux(fLnu_cmp) 
        
        return fLnu_cmp
            





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    agn = agnsed(M=1e8, dist=100, log_mdot=-1, a=0, cos_inc=0.5,
                 kTe_hot=100, kTe_warm=0.2, gamma_hot=2.5, gamma_warm=2.7,
                 r_hot=10, r_warm=200, log_rout=-1, fcol=1, h_max=10, rep=True,
                 z=0)
    
    nus = agn.nu_grid
    Es = agn.Egrid
    
    
    agn.set_units('counts')
    fEs = agn.get_SED(as_flux=True)
    
    plt.loglog(Es, Es**2 * fEs, color='k')
    plt.ylabel('Energy flux density [keV/cm$^2$/s/keV]')
    plt.xlabel('Energy [keV]')
    plt.ylim(1e-4, 1)
    plt.show()
    
