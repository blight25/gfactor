# -*- coding: utf-8 -*-

# Default libraries
import warnings
from typing import List, Tuple

# Third-party libraries
import pandas as pd
import numpy as np
from astropy.units import Quantity, cds
from astropy import units as u

# Local libraries
from gfactor import LISIRDRetriever, NISTRetriever
from gfactor.main.gfactorsolar import SolarSpectrum
from gfactor.main.gfactoratomic import AtomicData

from matplotlib import pyplot as plt


# Filter warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class gfactor:

    # UNITS
    c_cm = cds.c.cgs # speed of light in cm/s
    Quantum_const = 2.8179398e-13*u.cm
    k = 8.61733326e-5*u.eV / u.K  # Boltzmann constant eV/K
    h = 6.62606957e-27*u.erg*u.s
    

    def __init__(self):

        # Retrieval objects
        self._lisird_retriever = LISIRDRetriever()
        self._nist_retriever = NISTRetriever()

        # Data storage
        self._ion_id = []
        self._lines = []
        self._g_factors = []
        self._gf_dataframe = None


    @staticmethod
    def __joule_to_photons(spectrum):
        """
        Converts the energy flux in W/m^2 to a photon flux in cm^-2.

        Parameters:
        spectrum : Spectrum
            The input spectrum with energy flux in W/m^2.

        Returns:
        Spectrum
            The spectrum with photon flux in cm^-2.
        """

        # W/m^2/A to erg/cm^2/s/A
        spectrum = spectrum.with_flux_unit(unit=u.erg / u.cm**2 / u.s / u.AA)

        # erg/cm^2/s/A to photons/cm^2/s/A
        spectrum = spectrum.with_flux_unit(unit= u.photon / u.cm**2 / u.s / u.AA, 
                                            equivalencies=u.spectral_density(spectrum.spectral_axis))
        return spectrum
    

    @staticmethod
    def __extract_state_data(nist_table:pd.DataFrame, row_idx:int, line:Quantity, wavelength_bounds:Tuple[float]):
        """
        Extracts atomic data from NIST Dataframe for the current emission line and state.

        Parameters:
        nist_table : pd.DataFrame
            Atomic data from NIST.
        row_idx : int
            Row index to extract from the NIST table.
        line : Quantity
            Emission wavelength in Angstroms.
        wavelength_bounds: Tuple
            Emission wavelength bounds as (lower bound, upper bound) in Angstroms.

        Returns:
        dict
            Dictionary containing atomic transition data.
        """

        row = nist_table[row_idx]

        if row['term_k'] == 'nan':
            dataframe_multiplet = nist_table[(nist_table['element'] == row['element']) & (
                    nist_table['term_i'] == row['term_i'])]
        
        elif 3655*u.AA <= line <= 3972*u.AA and row['element'] == 'H':
            dataframe_multiplet = nist_table[
                (nist_table['element'] == row['element']) & (nist_table['obs_wl(A)'] >= 3655*u.AA) & (
                        nist_table['obs_wl(A)'] <= 3972*u.AA)]
        
        else:
            dataframe_multiplet = nist_table[(nist_table['element'] == row['element']) & (
                    nist_table['conf_i'] == row['conf_i']) & (
                                                         nist_table['conf_k'] == row['conf_k'])]
            
        a_val = row['Aki(s^-1)']  # Get relevant A_ik values
        E = row['Ei(eV)']  # E is eV
        E_k = row['Ek(eV)']  # E is eV
        jpp_val = row['J_i']  # Get momentum
        jpp = dataframe_multiplet['J_i']  # Check other momentum values
        ei = dataframe_multiplet['Ei(eV)']  # Get other energies for the state calculation
        f_row = row["fik"]
        constants_a = dataframe_multiplet['Aki(s^-1)']

        constants = {'Aki(s^-1)': a_val, 'Ei(eV)': E, 'Ek(eV)': E_k, 'J_i': jpp_val,
                     'J_i_other': jpp, 'Ei_other': ei, 'fik': f_row, 'Aki_other': constants_a,
                     'element': row['element'], 'sp_num': row['sp_num']}
        
        return constants
    

    def __get_state_prob(self, constants, T):
        """
        Calculates Boltzmann factors for all possible rotational energy states and determines the
        associated probability of the current state.

        Parameters:
        constants : dict
            Dictionary containing atomic transition data.
        T : float
            Temperature in Kelvin.

        Returns:
        float
            Current state probability.
        """

        states = (2 * constants['J_i_other'] + 1) * np.exp(constants['Ei_other'] / (gfactor.k * T))
        state = (2 * constants['J_i'] + 1) * np.exp(-constants['Ei(eV)'] / (gfactor.k * T))
        prob = state / sum(states)
        ion = str(constants['element']) + str(constants['sp_num'])
        self._ion_id.append(ion)
        return prob
    

    def __get_gfactors(self, nist_table: pd.DataFrame, spectrum: SolarSpectrum, T: float, hel_v: float, hel_d: float, 
                       wavelength_bounds:Tuple[float], debug_bounds=None):
        """
        Calculates gfactors for the given atomic transition data, solar spectrum, 
        temperature, and heliocentric conditions.

        Parameters:
        nist_table : pd.DataFrame
            Atomic data from NIST.
        spectrum : SolarSpectrum
            Solar spectrum object.
        T : float
            Temperature in Kelvin.
        hel_v : float
            Heliocentric velocity in cm/s.
        hel_d : float
            Heliocentric distance in AU.
        wavelength_bounds: Tuple
            Emission wavelength bounds as (lower bound, upper bound) in Angstroms.

        Returns:
        None
        """

        for i in range(len(nist_table['obs_wl(A)'])):

            # Emission line
            line = nist_table['obs_wl(A)'][i]
            
            # Check if current wavelength is within range of solar spectrum
            if line > spectrum.spectral_axis[-1]:
                raise ValueError('no comparable solar data for this bandpass: ' + str(nist_table['obs_wl(A)'].iloc[i]))   

            # Constants and transition probabilities
            constants = self.__extract_state_data(nist_table, i, line, wavelength_bounds)
            prob = self.__get_state_prob(constants, T)

            #Determine 99% velocity max assuming Boltzmann distribution of molecules (this is the most robust/physically correct)
            #mol_v= np.sqrt(3*k*T/m) #m is atomic weight, k is Boltzmann constant, T is temp in K
            #Use Boltzmann distribution to estimate max velocity for species 
            #Convert that velocity to a wavelength delta, D_lambda

            D_lambda = 0.25*u.AA #Angstroms
            line_shift = line.to(u.cm) * np.sqrt((1 + hel_v / gfactor.c_cm) / (1 - hel_v / gfactor.c_cm))
            bandpass = spectrum.spectral_axis[(spectrum.spectral_axis > (line_shift-D_lambda)) \
                                            & (spectrum.spectral_axis < (line_shift+D_lambda))]
            solar_flux = np.interp(bandpass, spectrum.spectral_axis, spectrum.flux) * (1*u.AU / hel_d)**2
            singular_flux = np.max(solar_flux)
            singular_wavelength = bandpass[np.argmax(solar_flux)]
            constants_a_sum = constants['Aki_other'].sum()

            try:
                gf = constants['Aki(s^-1)'] / constants_a_sum * gfactor.Quantum_const.to(u.AA) * line_shift ** 2 * \
                constants['fik'] * np.pi * singular_flux * prob
            except ZeroDivisionError:
                raise ZeroDivisionError(f"Looks like one of these got set to zero:\n{line}, {constants['Aki(s^-1)']}, "
                                        f"{sum(constants['Aki_other'])}, {constants['fik']}, {solar_flux}")
            
            if debug_bounds is not None:
                line_shift_AA = line_shift.to(u.AA)
                if (line_shift_AA.value > debug_bounds[0]) & (line_shift_AA.value < debug_bounds[1]):
                    fig, ax = plt.subplots()
                    feature_waves = spectrum.spectral_axis[(spectrum.spectral_axis >= debug_bounds[0]*u.AA) &
                                                           (spectrum.spectral_axis <= debug_bounds[1]*u.AA)]
                    feature_flux = spectrum.flux[(spectrum.spectral_axis >= debug_bounds[0]*u.AA) &
                                                 (spectrum.spectral_axis <= debug_bounds[1]*u.AA)]
                    ax.plot(feature_waves, feature_flux, label="Spectrum")
                    ax.vlines(x=line.value, label="Line", ymin=min(feature_flux.value), ymax=max(feature_flux.value), colors="mediumseagreen", linestyles="dashed")
                    ax.vlines(x=line_shift_AA.value, label="Line Shift", ymin=min(feature_flux.value), ymax=max(feature_flux.value), colors="orange", linestyles="dashed")
                    ax.vlines(x=singular_wavelength.value, label="Max", ymin=min(feature_flux.value), ymax=max(feature_flux.value), colors="purple", linestyles="dashed")
                    ax.legend()
                    plt.savefig(f"./gfactor/main/debug.png")

            self._lines.append(line)
            self._g_factors.append(gf)

            
            

    def gfactors(self, elements: List[str], wavelength_bounds=[800, 6000], date="2019-12-19", T=300, 
                 hel_d=1, hel_v=5, debug_bounds=None) -> pd.DataFrame:
        """
        Calculates gfactors for a given set of atomic species and heliocentric conditions.

        Parameters:
        elements : list of str
            List of elements to analyze, e.g., ['H', 'Na', 'S', ...].
        wavelength_bounds : list of int, optional
            Bounds of the form [lower bound, upper bound] in Angstroms. Default is [800, 6000].
        date : str, optional
            Date of observation in the form "YYYY-MM-DD". Default is "2019-12-19".
        T : float, optional
            Temperature in Kelvin. Default is 300.
        hel_d : float, optional
            Heliocentric distance in AU. Default is 1.
        hel_v : float, optional
            Heliocentric velocity in km/s. Default is 5.

        Returns:
        pd.DataFrame
            DataFrame containing elements, wavelengths, and associated gfactors.
        """
        
        # ****************************************** DATA RETRIEVAL ********************************************

        # 1. Fetch NIST data
        nist_table = AtomicData.load_nist(elements=elements)
        nist_table = nist_table[(wavelength_bounds[0]*u.AA <= nist_table['obs_wl(A)']) 
                                & (nist_table['obs_wl(A)'] <= wavelength_bounds[1]*u.AA)]


        # 2. Fetch daily spectrum data and cut off lower portion of low-res spectra (high-res takes priority for smaller wavelengths)
        low_res_daily, high_res_daily = SolarSpectrum.daily_spectrum(date=date, dataset="NNL")
        lower_bound_mask = low_res_daily.spectral_axis > high_res_daily.spectral_axis[0]
        low_res_daily = low_res_daily.mask_by_axis(lower_bound_mask)

        # 4. Load SUMER data
        sumer = SolarSpectrum.sumer_spectrum()
        
        # ************************************** SPECTRUM CONSTRUCTION ********************************************
        
        # 1. Build convolution kernel (1305 OI line basis)
        height, mean, std, pixel_std = high_res_daily.feature_fit(height=1e-5, mean=1302.5, std=.4,
                                                                  feature = [1301, 1303.5])

        # 2. Scale SUMER data to match NRL data
        sumer_scaled, _, _, _ = SolarSpectrum.daily_fit(poly_degree=5,
                                                   sumer=sumer,
                                                   daily_spec=high_res_daily,
                                                   gaussian_std=pixel_std,
                                                   fit="polynomial")
        

        # 3. Stitch together into a single spectrum
        spec_1 = SolarSpectrum.stitch(spec_left=sumer, spec_right=sumer_scaled,
                                                  priority="right", coverage=.01, max_res_percentile=0)
        
        spec_2 = SolarSpectrum.stitch(spec_left=spec_1, spec_right=high_res_daily,
                                      priority="left", coverage=.01, max_res_percentile=0)
        
        spectrum = SolarSpectrum.stitch(spec_left=spec_2, spec_right=low_res_daily,
                                        priority="left", coverage=.01, max_res_percentile=0)
        
        
        # **************************************** G-FACTOR CALCULATION ********************************************
        
        # Solar flux: W/m^2/A to phts/cm^2/s/A, velocity: km/s to cm/s
        spectrum = self.__joule_to_photons(spectrum)
        hel_v = hel_v * u.km / u.s
        hel_v = hel_v.to(unit=u.cm / u.s)
            
        # Calculate g-factors
        self.__get_gfactors(nist_table, spectrum, T*u.K, hel_v, hel_d*u.AU,
                            wavelength_bounds=wavelength_bounds, debug_bounds=debug_bounds)

        # Generate g-factor DataFrame
        gf_dataframe = pd.DataFrame(list(zip(self._ion_id, self._lines, self._g_factors)),
                                         columns=['Ion ID', 'Wavelength (Angstroms)', 'g-factor (phts s^-1)'])

        self._gf_dataframe = gf_dataframe
        return gf_dataframe
    

# Demo
if __name__ == "__main__":
    
    x = gfactor()
    gf_dataframe = x.gfactors(elements=['O'], date="2009-04-16", wavelength_bounds=(800, 4500), 
                              T=300, hel_d=.352, hel_v=0, debug_bounds=[1301, 1307]) 
    print(gf_dataframe)
