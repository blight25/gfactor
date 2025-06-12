from gfactor.querying.LISIRDQuerying import LISIRDRetriever

import numpy as np
import pandas as pd
import string
from typing import List, Dict
from pathlib import Path

from specutils.manipulation.resample import FluxConservingResampler
from specutils import Spectrum1D

from astropy.units import Quantity
from astropy.convolution import convolve, Gaussian1DKernel
import astropy.units as u

from scipy.special import legendre

from lmfit import Parameters, minimize

import time


class SolarSpectrum(Spectrum1D):

    """Extension of astropy's Spectrum1D for 
        1. data loading/querying
        2. spectral manipulation (stitching, resampling, convolution)
        3. spectral scaling (match static high-resolution data to dynamic low-resolution data)
        
        for more information on Spectrum1D, see https://specutils.readthedocs.io/en/stable/api/specutils.Spectrum1D.html
        """
    
    daily_retriever = LISIRDRetriever()
    daily_spectra = list(daily_retriever.irradiance_identifiers.keys())
    resampler = FluxConservingResampler()


    def __init__(self, name=None, emissions:Dict[str, List[float]]=None, *args, **kwargs):

        """Construct a SolarSpectrum Object: core arguments from Spectrum1D are
            1. flux, Quantity object (a.k.a array of flux values with units)
            2. spectral_axis, Quantity object (array of wavelength values with units)
            3. uncertainty, NDUncertainty object (infers from flux unit if no unit is given)

            For additional information, see https://specutils.readthedocs.io/en/stable/api/specutils.Spectrum1D.html
            Note that additional attributes may not be maintained by the current methods.

            Parameters
            ----------
            name: str
                identifier for the spectrum
            emissions: dict
                optional dictionary of emission features - see 'SolarSpectrum.emissions' for details
            """

        super().__init__(*args, **kwargs)
        
        self._name = name # Useful for distinguishing between SolarSpectrum objects

        # Global resolution - slight variation from this value over smaller windows is expected
        self.global_res = (self.spectral_axis[-1] - self.spectral_axis[0]) / len(self.spectral_axis)  

        # Emissions: maps feature identifiers (e.g. Lyman-Alpha, CI 1335, etc.) to wavelength bounds, 
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
        }

        Parameters
        ----------
        new_emissions : Dict[str, List[float]]
            Dictionary of emission features to initialize or overwrite.

        Returns
        -------
        None
        """

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
        }

        Parameters
        ----------
        new_emissions : Dict[str, List[float]]
            Dictionary of new emission features to add.

        Returns
        -------
        None
        """
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
        BASS 2000 is not queryable, so there is no fail-safe if this file is not identifiable.

        Parameters
        ----------
        sumer_file : str
            Exact filepath for accessing SUMER.
        emissions : Dict[str, List[float]], optional
            Dictionary of emission features - see 'SolarSpectrum.emissions' for details.

        Returns
        -------
        spectrum : SolarSpectrum
            SUMER spectrum as a SolarSpectrum object
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
                       daily_dir:str=None):
        
        """ Loads in SolarSpectrum object(s) specified by a given dataset and date. Currently 
        available datasets are as follows (see descriptions from UC Boulder's 
        LISIRD: https://lasp.colorado.edu/lisird/):

        1. {NNL: {high-res: 'https://lasp.colorado.edu/lisird/data/nnl_ssi_P1D'}
                 {low-res: 'https://lasp.colorado.edu/lisird/data/nnl_hires_ssi_P1D'} }

        2. {SORCE: {high-res: 'https://lasp.colorado.edu/lisird/data/sorce_solstice_ssi_high_res'}
                   {low-res: 'https://lasp.colorado.edu/lisird/data/sorce_ssi_l3'} }       
                 
        3. {TIMED: 'https://lasp.colorado.edu/lisird/data/timed_see_ssi_l3'}

        For spectra with both low and high resolution designations, both versions will be loaded
        and returned as tuple of SolarSpectrum objects (low-res, high-res): you can always
        check the '.name' attribute to confirm which is which. Note that the chosen date may not always 
        be valid for both subsets - in this case, the unavailable subset will be set to None.

        If there are no such designations, then the first tuple value will be a SolarSpectrum, 
        and the second will be None.

        Parameters
        ----------
        date : str
            Date for the spectrum.
        dataset : str, optional
            String dataset name - can be lower or uppercase.
        emissions : Dict[str, List[float]], optional
            Dictionary of emission features - see 'SolarSpectrum.emissions' for details.
        daily_dir : str, optional
            Specifies where to search for existing solar files. If no match is found, data
            will be queried from LISIRD directly (this may take a few seconds).

        Returns
        -------
        specs : Tuple[Optional[SolarSpectrum], Optional[SolarSpectrum]]
            Low-resolution and high-resolution Solar Spectrum objects (subject to the caveats
            listed above).
        """
        dataset = dataset.upper()
        if dataset not in SolarSpectrum.daily_spectra:
            raise ValueError(f"{dataset} not a supported daily spectrum: "
                             f"Currently supported spectra (queried from LISIRD " 
                             f"Database @ https://lasp.colorado.edu/lisird/) "
                             f"are {SolarSpectrum.daily_spectra}.")
        
        # Path object
        daily_dir = Path(daily_dir + "/" + dataset) if daily_dir else None

        # Dataset subsets
        subsets = SolarSpectrum.daily_retriever.irradiance_identifiers # Get subsets
        specs = []
        for subset in subsets[dataset]:
            if subset:
                file = daily_dir / date / f"{dataset}_{subset}.pickle" if daily_dir else None # Filename
            else:
                file = daily_dir / date / f"{dataset}.pickle" if daily_dir else None # Filename

            # Read internally
            if file and file.exists():
                data = pd.read_pickle(file)
            # Query 
            else:
                if file:
                    print(f"file {file} could not be found: querying from LISIRD directly")
                data = SolarSpectrum.daily_retriever.retrieve(dataset=dataset,
                                                          subset=subset,
                                                          query_date=date,
                                                          timeout=10)
        
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
    def resample(spec: 'SolarSpectrum', new_axis: Quantity):
        """
        Resamples the flux from a given spectrum's solar axis onto a new axis using astropy's FluxConservingResampler.

        Parameters
        ----------
        spec : SolarSpectrum
            The spectrum to be resampled.
        new_axis : Quantity
            The new spectral axis for resampling. Units must match the units of `spec.spectral_axis`.

        Returns
        -------
        resampled_spectra : SolarSpectrum
            A new SolarSpectrum object with the flux resampled onto the new spectral axis.
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
    
    
    def _resample(self, new_axis: Quantity):
        """
        Resamples the flux of the current spectrum in-place onto a new spectral axis using astropy's FluxConservingResampler.

        Parameters
        ----------
        new_axis : Quantity
            The new spectral axis for resampling. Units must match the units of `self.spectral_axis`.

        Returns
        -------
        None
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
    def convolution(spec: 'SolarSpectrum', std: float = 1):
        """
        Performs a Gaussian convolution on the given spectrum's flux and returns a new SolarSpectrum object.

        Parameters
        ----------
        spec : SolarSpectrum
            The spectrum to be convolved.
        std : float, optional
            Standard deviation for the Gaussian kernel. Default is 1.

        Returns
        -------
        convolved_spectra : SolarSpectrum
            A new SolarSpectrum object with convolved flux.
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
    
    
    def _convolution(self, std: float = 1):
        """
        Performs a Gaussian convolution in-place on the flux of the current spectrum.

        Parameters
        ----------
        std : float, optional
            Standard deviation for the Gaussian kernel. Default is 1.

        Returns
        -------
        None
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
    def stitch(spec_left: 'SolarSpectrum', spec_right: 'SolarSpectrum', priority: str = "left",
               coverage: float = 0.10, max_res_percentile: float = 0.05, return_stitch_points: bool = False):
        """
        Stitches two spectra together, prioritizing one over the other in overlapping regions.

        Parameters
        ----------
        spec_left : SolarSpectrum
            The left spectrum to stitch.
        spec_right : SolarSpectrum
            The right spectrum to stitch.
        priority : str, optional
            Priority for overlapping regions ('left' or 'right'). Default is 'left'.
        coverage : float, optional
            Fraction of overlap to consider for stitching. Default is 0.10.
        max_res_percentile : float, optional
            Maximum residual percentile for filtering. Default is 0.05.
        return_stitch_points : bool, optional
            Whether to return stitch points. Default is False.

        Returns
        -------
        stitched_spectrum : SolarSpectrum
            The stitched spectrum.
        stitch_points : Optional[List[float]]
            Stitch points if `return_stitch_points` is True.
        """
        
        # Check flux unit compatability
        if spec_left.flux.unit != spec_right.flux.unit:
            raise ValueError(f"Left spectra's flux units of '{spec_left.flux.unit}' "
                             f"are incompatible with right spectra's "
                             f"flux units '{spec_right.flux.unit}'")

        # Check wavelength unit compatability
        if spec_left.spectral_axis.unit != spec_right.spectral_axis.unit:
            raise ValueError(f"Left spectra's wavelength units of '{spec_left.spectral_axis.unit}' "
                             f" are incompatible with right spectra's "
                             f" wavelength units '{spec_right.spectral_axis.unit}'")

        # Check hyperparameter values
        if not 0 <= coverage <= 1:
            raise ValueError("'coverage' must be a float between 0 and 1")
    
        if not 0 <= max_res_percentile <= 1:
            raise ValueError("'max_residual' must be a float between 0 and 1")

        # Preserve spec_left
        if priority == "left":

            # Array of indices
            wave_idxs = np.array(list(range(len(spec_left.spectral_axis))))

            # Overlap mask: retain only the region of overlap
            overlap_mask = spec_left.spectral_axis >= spec_right.spectral_axis[0]
            overlap_idxs = wave_idxs[overlap_mask]

            # Coverage mask: filters out the first (1 - coverage)*100% of the overlapping data
            coverage_mask = (overlap_idxs - overlap_idxs[0] / (len(overlap_idxs) - 1)) >= (1 - coverage)
            coverage_idxs = overlap_idxs[coverage_mask]

            # Filter left spectrum down to the 'coverage' region
            spec_left_waves = spec_left.spectral_axis[coverage_idxs]
            spec_left_flux = spec_left.flux[coverage_idxs]

            # Interpolate right spectrum in the 'coverage' region 
            spec_right_flux = np.interp(spec_left_waves, spec_right.spectral_axis, spec_right.flux)

            # Flux residuals in 'coverage' region
            flux_diffs = np.absolute(spec_right_flux - spec_left_flux)

            # Obtains the max_residual*100% percentile of the residuals
            diff_percentile = np.percentile(flux_diffs, max_res_percentile*100)

            # Filter out all candidate wavelengths with residuals greater than diff_percentile
            stitch_waves = spec_left_waves[flux_diffs < diff_percentile]

            # Take the last candidate available (if max_residual = 0, just take left spectrum's upper wavelength bound)
            stitch_wave = stitch_waves[-1] if len(stitch_waves) > 0 else spec_left_waves[-1]

            # Stitch wavelengths and flux at the candidate wavelength
            combined_waves = np.concatenate((spec_left.spectral_axis[spec_left.spectral_axis <= stitch_wave], 
                                            spec_right.spectral_axis[spec_right.spectral_axis > stitch_wave]))
            combined_flux = np.concatenate((spec_left.flux[spec_left.spectral_axis <= stitch_wave], 
                                            spec_right.flux[spec_right.spectral_axis > stitch_wave]))
            
            # Stitch uncertainty, if provided
            if spec_left.uncertainty and spec_right.uncertainty:
                combined_uncertainty = np.concatenate(spec_left.uncertainty[spec_left.spectral_axis <= stitch_wave], spec_right.uncertainty[spec_right.spectral_axis > stitch_wave])
            else:
                combined_uncertainty = None


        # Preserve spec_right
        else:

            # Array of indices
            wave_idxs = np.array(list(range(len(spec_right.spectral_axis))))

            # Overlap mask: retain only the region of overlap
            overlap_mask = spec_right.spectral_axis <= spec_left.spectral_axis[-1]
            overlap_idxs = wave_idxs[overlap_mask]

            # Coverage mask: filters out the last (1 - coverage)*100% of the overlapping data
            coverage_mask = (overlap_idxs - overlap_idxs[0]) / (len(overlap_idxs) - 1) <= coverage
            coverage_idxs = overlap_idxs[coverage_mask]

            # Filter right spectrum down to the 'coverage' region
            spec_right_waves = spec_right.spectral_axis[coverage_idxs]
            spec_right_flux = spec_right.flux[coverage_idxs]

            # Interpolate left spectrum in the 'coverage' region 
            spec_left_flux = np.interp(spec_right_waves, spec_left.spectral_axis, spec_left.flux)

            # Flux residuals in 'coverage' region
            flux_diffs = np.absolute(spec_left_flux - spec_right_flux)

            # Obtains the max_residual*100% percentile of the residuals
            diff_percentile = np.percentile(flux_diffs, max_res_percentile*100)

            # Filter out all candidate wavelengths with residuals greater than diff_percentile
            stitch_waves = spec_right_waves[flux_diffs < diff_percentile]

            # Take the first candidate available (if max_residual = 0, just take right spectrum's lower wavelength bound)
            stitch_wave = stitch_waves[0] if len(stitch_waves) > 0 else spec_right_waves[0]

            # Stitch wavelengths and flux at the candidate wavelength
            combined_waves = np.concatenate((spec_left.spectral_axis[spec_left.spectral_axis < stitch_wave], spec_right.spectral_axis[spec_right.spectral_axis >= stitch_wave]))
            combined_flux = np.concatenate((spec_left.flux[spec_left.spectral_axis < stitch_wave], spec_right.flux[spec_right.spectral_axis >= stitch_wave]))
            
            # Stitch uncertainty, if provided
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

        # Wrap with SolarSpectrum
        stitched_spectrum = SolarSpectrum(flux=combined_flux, spectral_axis=combined_waves, 
                                          uncertainty=combined_uncertainty, emissions=emissions)
        
        # Returns all candidate stitch points, if requested
        if return_stitch_points:
            return stitched_spectrum, stitch_waves
        # Returns spectrum by itself
        else:
            return stitched_spectrum
       
        
    @staticmethod
    def spectral_overlap(spec1: 'SolarSpectrum', spec2: 'SolarSpectrum'):
        """
        Finds the overlapping region of two solar spectra and returns versions of each spectrum
        bounded by the overlap region. The left and right spectra are identified automatically.

        Parameters
        ----------
        spec1 : SolarSpectrum
            The first SolarSpectrum object.
        spec2 : SolarSpectrum
            The second SolarSpectrum object. Units between `spec1` and `spec2` must be compatible.

        Returns
        -------
        spec1_overlap : SolarSpectrum
            A SolarSpectrum object of `spec1` bounded by the overlap region.
        spec2_overlap : SolarSpectrum
            A SolarSpectrum object of `spec2` bounded by the overlap region.
        """

        # Check flux unit compatability
        if spec1.flux.unit != spec2.flux.unit:
            raise ValueError(f"Left spectra's Flux units of '{spec1.flux.unit}' " 
                             f" are incompatible with right spectra's flux units '{spec2.flux.unit}'")
        
        # Check wavelength unit compatability
        if spec1.spectral_axis.unit != spec2.spectral_axis.unit:
            raise ValueError(f"Left spectra's wavelength units of '{spec1.spectral_axis.unit}' "
                             f" are incompatible with right spectra's wavelength units '{spec2.spectral_axis.unit}'")

        # Masks for trimming
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
        
        # Apply appropriate masking to spec1
        waves1_overlap = spec1.spectral_axis[(trim1[0]) & (trim1[1])]
        flux1_overlap = spec1.flux[(trim1[0]) & (trim1[1])]

        # Applay appropriate masking to spec2
        waves2_overlap = spec2.spectral_axis[(trim2[0]) & (trim2[1])]
        flux2_overlap = spec2.flux[(trim2[0]) & (trim2[1])]

        # Carry over emissions
        emissions = {}
        for key, value in list(spec1.emissions.items()):
            emissions[key] = value['Wavelengths']
        for key, value in list(spec2.emissions.items()):
            emissions[key] = value['Wavelengths']
        
        emissions = emissions if len(emissions) != 0 else None

        # Trimmed spectra
        spec1_overlap = SolarSpectrum(flux=flux1_overlap, spectral_axis=waves1_overlap, emissions=emissions)
        spec2_overlap = SolarSpectrum(flux=flux2_overlap, spectral_axis=waves2_overlap, emissions=emissions)

        return spec1_overlap, spec2_overlap


    def integrated_flux(self, bounds: List[float] = None, return_res: bool = False):
        """
        Calculates the integrated flux for the spectrum over a specified emission feature.

        Parameters
        ----------
        bounds : List[float], optional
            Wavelength bounds in the form [lower bound, upper bound]. If None, calculates the
            integrated flux of the entire spectrum.
        return_res : bool, optional
            If True, also returns the resolution over the feature.

        Returns
        -------
        integrated_flux : float
            The integrated flux over the specified bounds or the entire spectrum.
        res : float, optional
            The resolution over the feature, returned only if `return_res` is True.
        """
        
        # Specific feature
        if bounds is not None:

            # Convert units
            if not isinstance(bounds, Quantity):
                bounds = bounds * self.spectral_axis.unit

            # Check feature bound validity
            if bounds[0] < self.spectral_axis[0] or bounds[-1] > self.spectral_axis[-1]:
                raise ValueError("feature bounds must fall within range of spectral axis")
            
            # Wavelengths, flux, resolution over emission feature
            waves = self.spectral_axis[(self.spectral_axis >= bounds[0]) & (self.spectral_axis <= bounds[1])]
            flux = self.flux[(self.spectral_axis >= bounds[0]) & (self.spectral_axis <= bounds[1])]
            res = (waves[-1] - waves[0]) / len(waves)
        
        # Whole spectrum
        else:
            flux = self.flux
            res = (self.spectral_axis[-1] - self.spectral_axis[0]) / len(self.spectral_axis)
        
        if return_res:
            return np.sum(flux * res), res
        else:
            return np.sum(flux * res)


    """ --------------------------------------------------- FEATURE FITTING -------------------------------------------------- """


    @staticmethod
    def _gaussian_func(params, feature_waves, feature_flux):
        """
        Gaussian fit function for feature fitting with the lmfit package.

        Parameters
        ----------
        params : lmfit.Parameters
            The parameters for the Gaussian fit.
        feature_waves : Quantity
            The wavelengths of the feature to fit.
        feature_flux : Quantity
            The flux values of the feature to fit.

        Returns
        -------
        residuals : np.ndarray
            The residuals between the model and the feature flux.
        """
        vals = params.valuesdict()
        model = vals['height'] * np.exp(-0.5 * np.square((feature_waves.value - vals['mean']) / vals['std']))
        return model - feature_flux.value


    def feature_fit(self, feature: List[float], height: float = 1, mean: float = 0, std: float = 1):
        """
        Fits a Gaussian to a given emission feature using initial parameter guesses.

        Parameters
        ----------
        feature : List[float]
            Wavelength bounds in the form [lower bound, upper bound].
        height : float, optional
            Initial guess for the Gaussian height. Default is 1.
        mean : float, optional
            Initial guess for the Gaussian center. Default is 0.
        std : float, optional
            Initial guess for the Gaussian spread. Default is 1.

        Returns
        -------
        height : Quantity
            The fitted height of the Gaussian.
        mean : Quantity
            The fitted mean of the Gaussian.
        std : Quantity
            The fitted standard deviation of the Gaussian.
        pixel_std : float
            The standard deviation in pixel space for Gaussian kernel construction.
        """
        
        # Set up lmfit parameters
        params = Parameters()
        params.add("height", value=height, min=0)
        params.add("mean", value=mean, min=0)
        params.add("std", value=std, min=0)
        
        # Convert to spectral axis units
        if not isinstance(feature, Quantity):
            feature = feature*self.spectral_axis.unit

        # spectrum wavelengths and flux on the feature
        feature_waves = self.spectral_axis[(self.spectral_axis >= feature[0]) & (self.spectral_axis <= feature[-1])]
        feature_flux = self.flux[(self.spectral_axis >= feature[0]) & (self.spectral_axis <= feature[-1])]

        # Fitting procedure
        fit_results = minimize(self._gaussian_func, params, args=(feature_waves, feature_flux))
        
        # Fitted params
        height, mean, std = list(fit_results.params.valuesdict().values())
        height = height * self.flux.unit
        mean = mean * self.spectral_axis.unit
        std = std * self.spectral_axis.unit
        
        # Spectrum resolution over feature
        res = (feature_waves[-1] - feature_waves[0]) / len(feature_waves)
        pixel_std = std / res # Resolution in pixel space
        
        return height, mean, std, pixel_std
    

    """ --------------------------------------------------- SPECTRUM FITTING -------------------------------------------------- """

    @staticmethod
    def _poly_func(params, sumer:'SolarSpectrum', daily_spec:'SolarSpectrum', gaussian_std, regress: bool = False):
        """
        Polynomial fit function for spectrum fitting (SUMER to daily spectra) using lmfit.

        Parameters
        ----------
        params : lmfit.Parameters
            The parameters for the polynomial fit.
        sumer : SolarSpectrum
            The SUMER spectrum to fit.
        daily_spec : SolarSpectrum
            The daily spectrum to match.
        gaussian_std : Quantity
            The standard deviation for the Gaussian kernel, in pixel space.
        regress : bool, optional
            If True, used for training. If False, used for evaluation. Default is False.

        Returns
        -------
        output : SolarSpectrum
            The model output (SUMER scaled).
        downsampled_output : SolarSpectrum
            The output after downsampling.
        dc_output : SolarSpectrum
            The output after downsampling and convolution.
        """
        
        # Coefficients
        coeffs = params.valuesdict()

        # f(x) = c0 + c1*x + c2*x**2 + ..., where x = SUMER.spectral_axis
        transformation = np.zeros_like(sumer.spectral_axis.value)
        for i, val in enumerate(list(coeffs.values())):
            transformation += val * sumer.spectral_axis.value**(len(params) - i - 1)
        
        # Scaled SUMER, g(x) = f(x) * SUMER.flux
        output_flux = transformation * sumer.flux

        # Wrap with SolarSpectrum
        output = SolarSpectrum(flux=output_flux, spectral_axis=sumer.spectral_axis)
        
        # Downsample onto daily spectrum wavelengths
        downsampled_output = SolarSpectrum.resample(spec=output, new_axis=daily_spec.spectral_axis)

        # Gaussian Convolution on downsampled spectra
        dc_output = SolarSpectrum.convolution(downsampled_output, std=gaussian_std.value)
        
        # Training
        if regress:
            return (dc_output.flux - daily_spec.flux).value
        
        # Evaluation
        else:
            return output, downsampled_output, dc_output


    @staticmethod
    def _legendre_func(params, sumer:'SolarSpectrum', daily_spec:'SolarSpectrum', gaussian_std, regress: bool = False):
        """
        Legendre fit function for spectrum fitting (SUMER to daily spectra) using lmfit.

        Parameters
        ----------
        params : lmfit.Parameters
            The parameters for the Legendre fit.
        sumer : SolarSpectrum
            The SUMER spectrum to fit.
        daily_spec : SolarSpectrum
            The daily spectrum to match.
        gaussian_std : Quantity
            The standard deviation for the Gaussian kernel, in pixel space.
        regress : bool, optional
            If True, used for training. If False, used for evaluation. Default is False.

        Returns
        -------
        output : SolarSpectrum
            The model output (SUMER scaled).
        downsampled_output : SolarSpectrum
            The output after downsampling.
        dc_output : SolarSpectrum
            The output after downsampling and convolution.
        """
        
        # Constructs Legendre polynomial of degree n
        def leg_n(x, n):
            leg = legendre(n)
            P_n = leg(x)
            return P_n
        
        # Coefficients
        coeffs = params.valuesdict()
        
        # f(x) = c0 + c1*x + c2*x**2 + ..., where x = SUMER.spectral_axis
        transformation = np.zeros_like(sumer.spectral_axis.value)
        for i, val in enumerate(list(coeffs.values())):
            transformation += val * leg_n(sumer.spectral_axis.value, i)

        # Scaled SUMER, g(x) = f(x) * SUMER.flux   
        output_flux = transformation * sumer.flux

        # Wrap with SolarSpectrum
        output = SolarSpectrum(flux=output_flux, spectral_axis=sumer.spectral_axis)
        
        # Downsample onto daily spectrum wavelengths
        downsampled_output = SolarSpectrum.resample(spec=output, new_axis=daily_spec.spectral_axis)

        # Gaussian Convolution on downsampled spectra
        dc_output = SolarSpectrum.convolution(downsampled_output, std=gaussian_std.value)
        
        # Training
        if regress:
            return (dc_output.flux - daily_spec.flux).value
        
        # Evaluation
        else:
            return output, downsampled_output, dc_output


    @staticmethod
    def daily_fit(sumer: 'SolarSpectrum', daily_spec: 'SolarSpectrum', gaussian_std: Quantity,
                  poly_degree: int = 6, fit: str = "polynomial"):

        """
        Finds a polynomial scaling transformation which fits SUMER to a daily spectrum.

        Parameters
        ----------
        sumer : SolarSpectrum
            SUMER SolarSpectrum object.
        daily_spec : SolarSpectrum
            Daily (NNL, SORCE, TIMED) daily spectrum.
        gaussian_std : Quantity
            Standard deviation (in pixel space) for Gaussian convolution. Identify a feature 
            on the daily spectrum and run 'feature_fit' first to obtain a reliable estimate.
        poly_degree : int, optional
            Degree of the transformation. Default is 6.
        fit : str, optional
            If "polynomial", uses standard polynomial fit function. Otherwise, uses a Legendre polynomial fit function. Default is "polynomial".

        Returns
        -------
        output : SolarSpectrum
            SUMER scaled to match the daily spectrum.
        downsampled_output : SolarSpectrum
            SUMER scaled and downsampled.
        dc_output : SolarSpectrum
            SUMER scaled, downsampled, and convolved. This is directly compared to the daily spectrum for training and visualization.
        fitted_params : lmfit.Parameters
            Fit coefficients for the polynomial transformation, f(x), which is applied
            to SUMER as g(x) = f(x) * SUMER.flux.
        """

        # Ensure that only region of overlap is considered
        overlap_tolerance = 1 * daily_spec.spectral_axis.unit
        runs_left = np.abs(sumer.spectral_axis[0] - daily_spec.spectral_axis[0]) > overlap_tolerance
        runs_right = np.abs(sumer.spectral_axis[-1] - daily_spec.spectral_axis[-1]) > overlap_tolerance
        if runs_left or runs_right:
            sumer, daily_spec = SolarSpectrum.spectral_overlap(sumer, daily_spec)

        # Check flux unit compatibility
        if sumer.flux.unit != daily_spec.flux.unit:
            raise ValueError(f"Inconsistent units: SUMER flux with units '{sumer.flux.unit}' is incompatible with " \
                             f"daily flux units of '{daily_spec.flux.unit}'")

        # Check wavelength unit compatibility
        if gaussian_std.unit != u.dimensionless_unscaled:
            raise ValueError("Standard deviation for constructing Gaussian kernel must be dimensionless, but " \
                             f"std was given with units of {gaussian_std.unit}: you may need to divide by the " \
                             "spectral resolution first.")

        # Create lmfit parameters
        alphabet = list(string.ascii_lowercase)
        params = Parameters()
        for i in range(poly_degree):
            params.add(alphabet[i], value=1)

        # Fit function to use
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

        return output, downsampled_output, dc_output, fitted_params


if __name__ == "__main__":

    # Get the absolute path to the current file (LISIRDQuerying.py)
    current_file = Path(__file__).resolve()

    # Get the package root (assuming this file is always in gfactor/querying/)
    package_root = current_file.parents[2]  # gfactor/
    
    # Directory to load data from
    dir = (package_root / "data" / "spectra").as_posix()

    sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
    nnl_low = SolarSpectrum.daily_spectrum(date="2023-04-03", dataset="NNL", daily_dir=None)
