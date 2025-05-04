import numpy as np
import pandas as pd

from pathlib import Path

from specutils.manipulation.resample import FluxConservingResampler
from specutils import Spectrum1D

from gfactor.querying.LISIRDQuerying import LISIRDRetriever

from astropy.nddata import VarianceUncertainty, InverseVariance
from astropy.units import Quantity
from astropy.convolution import convolve, Gaussian1DKernel
import astropy.units as u

from scipy.special import legendre

from lmfit import Parameters, minimize

import string

from typing import List, Dict, Tuple


class SolarSpectrum(Spectrum1D):
    
    daily_retriever = LISIRDRetriever()
    daily_spectra = ["TIMED", "SORCE", "NNL"]
    resampler = FluxConservingResampler()


    def __init__(self, name=None, emissions:Dict[str, List[float]]=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._name = name # Useful for distinguishing between SolarSpectrum objects

        # Global resolution - slight variation from this value over smaller windows is expected
        self.global_res = (self.spectral_axis[-1] - self.spectral_axis[0]) / len(self.spectral_axis)  

        # Maps feature names (e.g. Lyman-Alpha, CI 1335, etc.) to wavelength bounds, 
        # integrated fluxes, and local resolution
        self._emissions = {}
        if emissions:
            for emission in list(emissions.keys()):
                bounds = emissions[emission]
                if not isinstance(bounds, Quantity):
                    bounds = bounds*self.spectral_axis.unit
                if bounds[0] < self.spectral_axis[0] or bounds[-1] > self.spectral_axis[-1]:
                    continue
                self._emissions[emission] = {}
                self._emissions[emission]["Wavelengths"] = bounds
                integrated_flux, res = self.integrated_flux(bounds, return_res=True)
                self._emissions[emission]["Integrated Flux"] = integrated_flux
                self._emissions[emission]["Resolution"] = res

        # Rename non-static methods
        self.resample = self._resample
        self.convolution = self._convolution


    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name
    
    @property
    def emissions(self):
        return self._emissions
    
    @emissions.setter
    def emissions(self, new_emissions:Dict[str, List[float]]):

        """
        Initializes or overwrites existing emission features - please refer to 'add_emissions' if 
        you wish to add more while preserving any existing features. 

        Takes a dictionary of the form 
        emissions = {emission_name: [lower wavelength bound:upper wavelength bound]}.

        SolarSpectrum objects then calculate a local resolution (res) and integrated flux (i_flux)
        for each emission feature.

        Results are stored as a nested dictionary of the form 
        {emission_name: 
        {
        "Wavelengths": [lower wavelength bound, upper wavelength bound], 
        "Resolution": res, 
        "Integrated Flux": i_flux}
        }"""

        for emission in list(new_emissions.keys()):
                bounds = new_emissions[emission]
                if not isinstance(bounds, Quantity):
                    bounds = bounds*self.spectral_axis.unit # Convert units
                
                # Catch invalid boundary conditions
                if bounds[0] < self.spectral_axis[0]:
                    raise ValueError(f"bound {bounds[0]} for feature {emission} falls below "
                                      f"spectral axis minimum of {self.spectral_axis[0]}")
                elif bounds[-1] > self.spectral_axis[-1]:
                    raise ValueError(f"bounds {bounds} for feature {emission} ")
                
                # Create emission dict
                self._emissions[emission] = {}
                self._emissions[emission]["Wavelengths"] = bounds
                integrated_flux, res = self.integrated_flux(bounds, return_res=True)
                self._emissions[emission]["Integrated Flux"] = integrated_flux
                self._emissions[emission]["Resolution"] = res
    

    def add_emissions(self, new_emissions:Dict[str, List[float]]):

        """
        Extends current emission features to include new emissions - preserves existing emissions.

        Takes a dictionary of the form 
        emissions = {emission_name: [lower wavelength bound:upper wavelength bound]}.

        SolarSpectrum objects then calculate a local resolution (res) and integrated flux (i_flux)
        for each emission feature.

        Results are stored as a nested dictionary of the form 
        {emission_name: 
        {
        "Wavelengths": [lower wavelength bound, upper wavelength bound], 
        "Resolution": res, 
        "Integrated Flux": i_flux}
        }"""

        for emission in list(new_emissions.keys()):
                bounds = new_emissions[emission]
                if not isinstance(bounds, Quantity):
                    bounds = bounds*self.spectral_axis.unit # Convert units
                
                # Catch invalid boundary conditions
                if bounds[0] < self.spectral_axis[0]:
                    raise ValueError(f"bound {bounds[0]} for feature {emission} falls below "
                                      f"spectral axis minimum of {self.spectral_axis[0]}")
                elif bounds[-1] > self.spectral_axis[-1]:
                    raise ValueError(f"bounds {bounds} for feature {emission} ")

                # Create emission dict
                self._emissions[emission] = {}
                self._emissions[emission]["Wavelengths"] = bounds
                integrated_flux, res = self.integrated_flux(bounds, return_res=True)
                self._emissions[emission]["Integrated Flux"] = integrated_flux
                self._emissions[emission]["Resolution"] = res


    @staticmethod
    def sumer_spectrum(sumer_file:str="./SUMER/SUMER.txt", emissions:Dict[str, List[float]]=None):

        """ Loads the SUMER spectrum and returns it as a Solar Spectrum object. SUMER is compiled from the
        BASS 2000 solar archive (https://bass2000.obspm.fr/solar_spect.php), and represents our 
        best high-resolution capture of solar activity between ~670 and ~1609 Angstroms. Unfortunately,
        BASS 2000 is not queryable, so there is no failsafe if this file is not identifiable.
        
        @param sumer_file: exact filepath for accessing SUMER
        
        @param emissions: optional dictionary of emission features - see 'SolarSpectrum.emissions' for details

        @return spectrum: SUMER spectrum as a SolarSpectrum object
         """

        # Read in SUMER file
        sumer = pd.read_csv(sumer_file)
    
        # Convert to numpy arrays
        wavelengths = sumer["Wavelength (Angstroms)"].values
        flux = sumer[" Normalized intensity"].values
        
        # Remove duplicates (if they exist) and convert units 
        # (Divide by 1000 to get from W to mW and multiply
        # by solid angle of Sun at 1 AU to get W/m^2/Angstrom)
        wavelengths, indices = np.unique(wavelengths, return_index=True)
        flux = flux[indices] * 6.794e-5 / 1000

        # Will include uncertainty here at some point: should be either VarianceUncertainty(Quantity) or StdDevUncertainty(Quantity)
        spectrum = SolarSpectrum(flux=Quantity(flux, unit=u.W / u.m**2 /u.AA), 
                                   spectral_axis=Quantity(wavelengths, unit=u.AA), 
                                   emissions=emissions)
        
        return spectrum
    

    @staticmethod
    def daily_spectrum(date:str, dataset:str="NNL",
                       emissions:Dict[str, List[float]]=None,
                       daily_dir:str="./data/spectra"):
        
        """ Loads in SolarSpectrum object(s) specified by a given dataset and date. Currently 
        available datasets are as follows (see descriptions from UC Boulder's 
        LISIRD: https://lasp.colorado.edu/lisird/):

        1. {NNL: {high-res: https://lasp.colorado.edu/lisird/data/nnl_ssi_P1D}
                 {low-res: https://lasp.colorado.edu/lisird/data/nnl_hires_ssi_P1D} }

        2. {SORCE: {high-res: https://lasp.colorado.edu/lisird/data/sorce_solstice_ssi_high_res}
                   {low-res: https://lasp.colorado.edu/lisird/data/sorce_ssi_l3} }       
                 
        3. {TIMED: https://lasp.colorado.edu/lisird/data/timed_see_ssi_l3}

        
        For spectra with both high and low resolution designations, both versions will be loaded
        and returned as tuple of SolarSpectrum objects (high-res, low-res): can always
        check '.name' attribute to confirm which is which. Note that the chosen date may not always 
        be valid for both subsets - in this case, the unavailable subset will be set to None.
        
        Otherwise, the first tuple value will be a SolarSpectrum, and the second will be None.

        @param dataset: string dataset name - can be lower or uppercase
        
        @param emissions: optional dictionary of emission features - see 'SolarSpectrum.emissions' for details
        
        @param daily_dir: specifies where to search for existing solar files. If no match is found, data
        will be queried from LISIRD directly (this may take a few seconds).

        @return specs = (spec_1, spec_2): high-resolution and low-resolution Solar Spectrum objects (subject to the caveats
        listed above).

        """
        dataset = dataset.upper()
        if dataset not in SolarSpectrum.daily_spectra:
            raise ValueError(f"{dataset} not a supported daily spectrum: "
                             f"Currently supported spectra (queried from LISIRD " 
                             f"Database @ https://lasp.colorado.edu/lisird/) "
                             f"are {SolarSpectrum.daily_spectra}.")
        
        # Path object
        daily_dir = Path(daily_dir + "/" + dataset)

        # Dataset subsets
        subsets = SolarSpectrum.daily_retriever.dataset_names() # Get subsets
        specs = []
        for subset in subsets[dataset]:

            file = daily_dir / date / subset + ".pickle" # Filename

            # Read internally
            if file.exists():
                data = pd.read_pickle(file)
            # Query 
            else:
                data = SolarSpectrum.daily_retriever.retrieve(dataset=dataset,
                                                          subset=subset,
                                                          date=date, 
                                                          max_queries=1)
        
            # Unit Conversion
            wavelengths = data["wavelength (nm)"] * 10
            flux = data["irradiance (W/m^2/nm)"] / 10

            # Spectrum object
            name = dataset + "_" + subset if subset else dataset
            spectrum = SolarSpectrum(flux=Quantity(flux, unit=u.W / u.m**2 /u.AA), 
                                    spectral_axis=Quantity(wavelengths, unit=u.AA),
                                    emissions=emissions,
                                    name=name)
            specs.append(spectrum)

        return tuple(specs)


    @staticmethod
    def resample(spec:'SolarSpectrum', new_axis:Quantity):

        """Resample the flux from a given spectrum's solar axis onto a
         new axis, using astropy's FluxConservingResampler. See 
         https://specutils.readthedocs.io/en/stable/api/specutils.manipulation.FluxConservingResampler.html
         for more details.
         
         @param spec: SolarSpectrum object
         @param new_axis: Quantity object, new axis for resampling. Note
         that the new axis unit must match the units of spec.spectral_axis.
         
         @return resampled_spectra: a new SolarSpectrum object, with
         spectral_axis = new_axis
         """
        
        # Check unit compatibility
        if spec.spectral_axis.unit != new_axis.unit:
            raise ValueError(f"Wavelength units of '{new_axis.unit}'  "
                             f" are incompatible with spectrum wavelength units "
                             f"'{spec.spectral_axis.unit}'")
        
        # Resampled spectra - but it's a Spectrum1D object
        resampled_spectra = SolarSpectrum.resampler(spec, new_axis)

        # If emissions exist, need to adjust values
        if len(spec.emissions) != 0:
            emission_keys = list(spec.emissions.keys())
            emission_dicts = list(spec.emissions.values())
            emission_bounds = [emission_dict["Wavelengths"] for emission_dict in emission_dicts]
            
            # SolarSpectrum object
            resampled_spectra = SolarSpectrum(flux=resampled_spectra.flux, 
                                spectral_axis=resampled_spectra.spectral_axis, 
                                emissions=dict(zip(emission_keys, emission_bounds)))
        
        # Otherwise, immediately wrap with SolarSpectrum
        else:
            resampled_spectra = SolarSpectrum(flux=resampled_spectra.flux, 
                                spectral_axis=resampled_spectra.spectral_axis)
        
        return resampled_spectra
    
    
    def _resample(self, new_axis):

        """Resamples self.flux in-place from self.spectral_axis onto a
         new axis, using astropy's FluxConservingResampler. See 
         https://specutils.readthedocs.io/en/stable/api/specutils.manipulation.FluxConservingResampler.html
         for more details.
         
         @param spec: SolarSpectrum object
         @param new_axis: Quantity object, new axis for resampling. Note
         that the new axis unit must match the units of spec.spectral_axis.
        
         """
        
        # Check unit compatability
        if self.spectral_axis.unit != new_axis.unit:
            raise ValueError(f"Wavelength units of '{new_axis.unit}' "
                             f" are incompatible with current wavelength units " 
                             f"'{self.spectral_axis.unit}'")

        resampled_spectra = SolarSpectrum.resampler(self, new_axis)

        # If emissions exist, need to adjust values
        if len(self._emissions) != 0:
            emission_keys = list(self._emissions.keys())
            emission_dicts = list(self._emissions.values())
            emission_bounds = [emission_dict["Wavelengths"] for emission_dict in emission_dicts]

            # SolarSpectrum object
            resampled_spectra = SolarSpectrum(flux=resampled_spectra.flux, 
                                spectral_axis=resampled_spectra.spectral_axis, 
                                emissions=dict(zip(emission_keys, emission_bounds)))
            
        # Otherwise, immediately wrap with SolarSpectrum
        else:
            resampled_spectra = SolarSpectrum(flux=resampled_spectra.flux, 
                                spectral_axis=resampled_spectra.spectral_axis)
        
        # In-place update
        self.__dict__.update(resampled_spectra.__dict__)
        
    
    @staticmethod
    def convolution(spec:'SolarSpectrum', std=1):

        """Performs a Gaussian convolution on the given spectrum's flux
         and returns it on the same spectral axis as a new SolarSpectrum object.
        
         @param spec: spectrum to be convolved
         @param std: standard deviation for the Gaussian kernel

         @return convolved_spectra: SolarSpectrum with convolved flux
          """
        
        kernel = Gaussian1DKernel(stddev=std)
        convolved_flux = convolve(spec.flux, kernel=kernel)
        convolved_spectra = SolarSpectrum(flux=convolved_flux, spectral_axis=spec.spectral_axis)

        # If emissions exist, need to carry over values
        if len(spec.emissions) != 0:
            emission_keys = list(spec.emissions.keys())
            emission_dicts = list(spec.emissions.values())
            emission_bounds = [emission_dict["Wavelengths"] for emission_dict in emission_dicts]
            emissions = dict(zip(emission_keys, emission_bounds))
            convolved_spectra.emissions = emissions
    
        return convolved_spectra
    
    
    def _convolution(self, std=1):

        """Performs a Gaussian convolution in-place on self.flux.
    
        @param spec: spectrum to be convolved
        @param std: standard deviation for the Gaussian kernel

        """
    
        kernel = Gaussian1DKernel(stddev=std)
        convolved_flux = convolve(self.flux, kernel=kernel)
        convolved_spectra = SolarSpectrum(flux=convolved_flux, spectral_axis=self.spectral_axis)

        # If emissions exist, need to carry over values
        if len(self._emissions) != 0:
            emission_keys = list(self._emissions.keys())
            emission_dicts = list(self._emissions.values())
            emission_bounds = [emission_dict["Wavelengths"] for emission_dict in emission_dicts]
            emissions = dict(zip(emission_keys, emission_bounds))
            convolved_spectra.emissions = emissions
        
        # In-place update
        self.__dict__.update(convolved_spectra.__dict__)
        

    @staticmethod
    def stitch(spec_left:'SolarSpectrum', spec_right: 'SolarSpectrum', priority="left",
               coverage=.10, max_residual=.05, return_stitch_points=False):
        
        if spec_left.flux.unit != spec_right.flux.unit:
            raise ValueError(f"Left spectra's flux units of '{spec_left.flux.unit}' "
                             f"are incompatible with right spectra's "
                             f"flux units '{spec_right.flux.unit}'")

        if spec_left.spectral_axis.unit != spec_right.spectral_axis.unit:
            raise ValueError(f"Left spectra's wavelength units of '{spec_left.spectral_axis.unit}' "
                             f" are incompatible with right spectra's "
                             f" wavelength units '{spec_right.spectral_axis.unit}'")

        if not 0 <= coverage <= 1:
            raise ValueError("'coverage' must be a float between 0 and 1")
    
        if not 0 <= max_residual <= 1:
            raise ValueError("'max_residual' must be a float between 0 and 1")


        if priority == "left":
            wave_idxs = np.array(list(range(len(spec_left.spectral_axis))))
            overlap_idxs = wave_idxs[spec_left.spectral_axis >= spec_right.spectral_axis[0]]

            coverage_idxs = overlap_idxs[overlap_idxs - overlap_idxs[0] / (len(overlap_idxs) - 1) >= (1 - coverage)]
            spec_left_waves = spec_left.spectral_axis[coverage_idxs]
            spec_left_flux = spec_left.flux[coverage_idxs]

            spec_right_flux = np.interp(spec_left_waves, spec_right.spectral_axis, spec_right.flux)
            flux_diffs = np.absolute(spec_right_flux - spec_left_flux)
            diff_percentile = np.percentile(flux_diffs, max_residual*100)
            stitch_waves = spec_left_waves[flux_diffs < diff_percentile]
            stitch_wave = stitch_waves[-1] if len(stitch_waves) > 0 else spec_left_waves[-1]

            combined_waves = np.concatenate((spec_left.spectral_axis[spec_left.spectral_axis <= stitch_wave], 
                                            spec_right.spectral_axis[spec_right.spectral_axis > stitch_wave]))
            combined_flux = np.concatenate((spec_left.flux[spec_left.spectral_axis <= stitch_wave], spec_right.flux[spec_right.spectral_axis > stitch_wave]))
            
            if spec_left.uncertainty and spec_right.uncertainty:
                combined_uncertainty = np.concatenate(spec_left.uncertainty[spec_left.spectral_axis <= stitch_wave], spec_right.uncertainty[spec_right.spectral_axis > stitch_wave])
            else:
                combined_uncertainty = None

        else:
            wave_idxs = np.array(list(range(len(spec_right.spectral_axis))))
            overlap_idxs = wave_idxs[spec_right.spectral_axis <= spec_left.spectral_axis[-1]]
            
            coverage_idxs = overlap_idxs[(overlap_idxs - overlap_idxs[0]) / (len(overlap_idxs) - 1) <= coverage]

            spec_right_waves = spec_right.spectral_axis[coverage_idxs]
            spec_right_flux = spec_right.flux[coverage_idxs]

            spec_left_flux = np.interp(spec_right_waves, spec_left.spectral_axis, spec_left.flux)
            flux_diffs = np.absolute(spec_left_flux - spec_right_flux)
            diff_percentile = np.percentile(flux_diffs, max_residual*100)
            stitch_waves = spec_right_waves[flux_diffs < diff_percentile]

            stitch_wave = stitch_waves[0] if len(stitch_waves) > 0 else spec_right_waves[0]

            combined_waves = np.concatenate((spec_left.spectral_axis[spec_left.spectral_axis < stitch_wave], spec_right.spectral_axis[spec_right.spectral_axis >= stitch_wave]))
            combined_flux = np.concatenate((spec_left.flux[spec_left.spectral_axis < stitch_wave], spec_right.flux[spec_right.spectral_axis >= stitch_wave]))
            
            if spec_left.uncertainty and spec_right.uncertainty:
                combined_uncertainty = np.concatenate((spec_left.uncertainty[spec_left.spectral_axis < stitch_wave], spec_right.uncertainty[spec_right.spectral_axis >= stitch_wave]))
            else:
                combined_uncertainty = None
        
        # Combine and carry over emission values, if any
        emissions = {}
        for key, value in list(spec_left.emissions.items()):
            emissions[key] = value['Wavelengths']
        for key, value in list(spec_right.emissions.items()):
            emissions[key] = value['Wavelengths']
        emissions = emissions if len(emissions) != 0 else None


        # Stitched
        stitched_spectrum = SolarSpectrum(flux=combined_flux, spectral_axis=combined_waves, 
                                          uncertainty=combined_uncertainty, emissions=emissions)
        
        # Returns all candidate stitch points, if requested
        if return_stitch_points:
            return stitched_spectrum, stitch_waves
        # Returns spectrum by itself
        else:
            return stitched_spectrum
       
        
    @staticmethod
    def spectral_overlap(spec1, spec2:'SolarSpectrum'):

        if spec1.flux.unit != spec2.flux.unit:
            raise ValueError(f"Left spectra's Flux units of '{spec1.flux.unit}' " 
                             f" are incompatible with right spectra's flux units '{spec2.flux.unit}'")

        if spec1.spectral_axis.unit != spec2.spectral_axis.unit:
            raise ValueError(f"Left spectra's wavelength units of '{spec1.spectral_axis.unit}' "
                             f" are incompatible with right spectra's wavelength units '{spec2.spectral_axis.unit}'")


        trim1 = [np.ones_like(spec1.spectral_axis.value).astype(bool)]*2
        trim2 = [np.ones_like(spec2.spectral_axis.value).astype(bool)]*2

        # Determine which spectrum runs left
        if spec1.spectral_axis[0] <= spec2.spectral_axis[0]:
            trim1[0] = (spec1.spectral_axis >= spec2.spectral_axis[0])
        else:
            trim2[0] = (spec2.spectral_axis >= spec1.spectral_axis[0])
        
        # Determine which spectrum runs right
        if spec1.spectral_axis[-1] >= spec2.spectral_axis[-1]:
            trim1[1] = (spec1.spectral_axis <= spec2.spectral_axis[-1])
        else:
            trim2[1] = (spec2.spectral_axis <= spec1.spectral_axis[-1])
        
        waves1_overlap = spec1.spectral_axis[(trim1[0]) & (trim1[1])]
        flux1_overlap = spec1.flux[(trim1[0]) & (trim1[1])]

        emissions = {}
        for key, value in list(spec1.emissions.items()):
            emissions[key] = value['Wavelengths']
        for key, value in list(spec2.emissions.items()):
            emissions[key] = value['Wavelengths']
        
        emissions = emissions if len(emissions) != 0 else None

        spec1_overlap = SolarSpectrum(flux=flux1_overlap, spectral_axis=waves1_overlap, emissions=emissions)

        waves2_overlap = spec2.spectral_axis[(trim2[0]) & (trim2[1])]
        flux2_overlap = spec2.flux[(trim2[0]) & (trim2[1])]
        spec2_overlap = SolarSpectrum(flux=flux2_overlap, spectral_axis=waves2_overlap, emissions=emissions)

        return spec1_overlap, spec2_overlap


    def integrated_flux(self, bounds:List[float]=None, return_res=False):

        if bounds is not None:
            if not isinstance(bounds, Quantity):
                bounds = bounds * self.spectral_axis.unit
            if bounds[0] < self.spectral_axis[0] or bounds[-1] > self.spectral_axis[-1]:
                raise ValueError("feature bounds must fall within range of spectral axis")
            waves = self.spectral_axis[(self.spectral_axis >= bounds[0]) & (self.spectral_axis <= bounds[1])]
            flux = self.flux[(self.spectral_axis >= bounds[0]) & (self.spectral_axis <= bounds[1])]
            res = (waves[-1] - waves[0]) / len(waves)
        else:
            flux = self.flux
            res = (self.spectral_axis[-1] - self.spectral_axis[0]) / len(self.spectral_axis)
        
        if return_res:
            return np.sum(flux * res), res
        else:
            return np.sum(flux * res)



    """ --------------------------------------------------- FEATURE FITTING -------------------------------------------------- """


    def _gaussian_func(params, feature_waves, feature_flux):
    
        vals = params.valuesdict()
        model = vals['height']*np.exp(-.5*np.square((feature_waves.value - vals['mean']) / vals['std']))
        return model - feature_flux.value


    def feature_fit(self, feature:List[float], fit_func=_gaussian_func, height=1, mean=0, std=1):
        
        params = Parameters()
        params.add("height", value=height, min=0)
        params.add("mean", value=mean, min=0)
        params.add("std", value=std, min=0)

        if not isinstance(feature, Quantity):
            feature = feature*self.spectral_axis.unit

        feature_waves = self.spectral_axis[(self.spectral_axis >= feature[0]) & (self.spectral_axis <= feature[-1])]
        feature_flux = self.flux[(self.spectral_axis >= feature[0]) & (self.spectral_axis <= feature[-1])]
        fit_results = minimize(fit_func, params, args=(feature_waves, feature_flux))
        
        # Params
        height, mean, std = list(fit_results.params.valuesdict().values())
        height = height * self.flux.unit
        mean = mean * self.spectral_axis.unit
        std = std * self.spectral_axis.unit

        res = (feature_waves[-1] - feature_waves[0]) / len(feature_waves)
        pixel_std = std / res # Resolution in pixel space
        
        return height, mean, std, pixel_std
    

    """ --------------------------------------------------- SPECTRUM FITTING -------------------------------------------------- """


    def _poly_func(params, sumer:'SolarSpectrum', daily_spec:'SolarSpectrum', gaussian_std, regress=False):
        
        vals = params.valuesdict()

        transformation = np.zeros_like(sumer.spectral_axis.value)
        for i, val in enumerate(list(vals.values())):
            transformation += val * sumer.spectral_axis.value**(len(params) - i - 1)
        
        output_flux = transformation * sumer.flux
        output = SolarSpectrum(flux=output_flux, spectral_axis=sumer.spectral_axis)
        
        downsampled_output = SolarSpectrum.resample(spec=output, new_axis=daily_spec.spectral_axis)
        dc_output = SolarSpectrum.convolution(downsampled_output, std=gaussian_std.value)
        
        if regress:
            return (dc_output.flux - daily_spec.flux).value
        else:
            return output, downsampled_output, dc_output


    def _legendre_func(params, sumer:'SolarSpectrum', daily_spec:'SolarSpectrum', gaussian_std, regress=False):
        
        def leg_n(x, n):
            leg = legendre(n)
            P_n = leg(x)
            return P_n
        
        vals = params.valuesdict()
        
        transformation = np.zeros_like(sumer.spectral_axis.value)
        for i, val in enumerate(list(vals.values())):
            transformation += val * leg_n(sumer.spectral_axis.value, i)
            
        output_flux = transformation * sumer.flux
        output = SolarSpectrum(flux=output_flux, spectral_axis=sumer.spectral_axis)
        
        downsampled_output = SolarSpectrum.resample(spec=output, new_axis=daily_spec.spectral_axis)
        dc_output = SolarSpectrum.convolution(downsampled_output, std=gaussian_std.value)
        
        if regress:
            return (dc_output.flux - daily_spec.flux).value
        else:
            return output, downsampled_output, dc_output



    def daily_fit(poly_degree, sumer, daily_spec, gaussian_std, fit="polynomial"):

        # Ensure that only region of overlap is considered
        overlap_tolerance = 1 * daily_spec.spectral_axis.unit
        runs_left = np.abs(sumer.spectral_axis[0] - daily_spec.spectral_axis[0]) > overlap_tolerance
        runs_right = np.abs(sumer.spectral_axis[-1] - daily_spec.spectral_axis[-1]) > overlap_tolerance
        if runs_left or runs_right: 
            sumer, daily_spec = SolarSpectrum.spectral_overlap(sumer, daily_spec)
        
        # Ensure units are consistent
        if sumer.flux.unit != daily_spec.flux.unit:
            raise ValueError(f"Inconsistent units: SUMER flux with units '{sumer.flux.unit}' is incompatible with " \
                             f"daily flux units of '{daily_spec.flux.unit}")
        
        if gaussian_std.unit != u.dimensionless_unscaled:
            raise ValueError("Standard deviation for constructing gaussian kernel must be dimensionless, but " \
                             f"std was given with units of {gaussian_std.unit}:\nyou may need to divide by the " \
                             "spectral resolution first.")

        # Create parameters
        alphabet = list(string.ascii_lowercase)
        params = Parameters()
        for i in range(poly_degree):
            params.add(alphabet[i], value=1)

        if fit == "polynomial":
            fit_func = SolarSpectrum._poly_func
        else:
            fit_func = SolarSpectrum._legendre_func

        # Obtain fit parameters
        fit_results = minimize(fit_func, params, args=(sumer, daily_spec, gaussian_std, True), method="least_squares")
        fitted_params = fit_results.params

        # Create fitted spectrum
        output, downsampled_output, dc_output = fit_func(fitted_params, 
                                                         sumer, 
                                                         daily_spec, 
                                                         gaussian_std, 
                                                         regress=False)
        
        # Update emissions
        emissions = {}
        for key, value in list(sumer.emissions.items()):
            emissions[key] = value['Wavelengths']
        for key, value in list(sumer.emissions.items()):
            emissions[key] = value['Wavelengths']
        
        if len(emissions) != 0:
            output.emissions = emissions
            downsampled_output.emissions = emissions
            dc_output.emissions = emissions
        
        return output, downsampled_output, dc_output, fit_results


if __name__ == "__main__":
    sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
    nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high", emissions={"Source: I made it up": [1212, 1220],                                                                                      "Lyman-alpha":[1214-1218]})
    feature = [1301, 1303.5]
    height, mean, pixel_std = nnl.feature_fit(feature, height=5e-5, mean=1302.2, std=.4)
    output, downsampled, dc, fit_results = SolarSpectrum.daily_fit(poly_degree=5, 
                                                            sumer=sumer,
                                                            daily_spec=nnl, 
                                                            gaussian_std = pixel_std,
                                                            fit='polynomial')
        
