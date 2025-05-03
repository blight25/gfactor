import pytest
import datetime as dt
import random
import matplotlib
matplotlib.use('Agg')

from astropy import units as u
from astropy.units import Quantity
from astropy.nddata import VarianceUncertainty, InverseVariance

import numpy as np

# Local imports
from gfactor.main.gfactorsolar import SolarSpectrum

def test_object():
    try:
        spectrum_object = SolarSpectrum(emissions={"Lyman-alpha": [1214, 1218], "Source: I made it up": [1212, 1220]}, 
                                        flux=[10, 10, 15, 10]*(u.W/u.m**2/u.Angstrom), 
                                        spectral_axis=[1210, 1214, 1218, 1222]*u.Angstrom,
                                        uncertainty=VarianceUncertainty([2, 2, 2, 2]*u.W**2/u.m**4/u.Angstrom**2))
        assert isinstance(spectrum_object.emissions, dict)
    except ValueError:
        assert False


def test_load_sumer():
    try:
        sumer = SolarSpectrum.sumer_spectrum()
        assert isinstance(sumer, SolarSpectrum)
    except ValueError:
        assert False

def test_load_daily():
    try:
        nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high")
        assert isinstance(nnl, SolarSpectrum)
    except ValueError:
        assert False


def test_spectral_overlap():
    try:
        sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
        nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high")
        sumer_overlap, nnl_overlap = SolarSpectrum.spectral_overlap(sumer, nnl)
        assert (np.abs(sumer_overlap.spectral_axis[0] - nnl_overlap.spectral_axis[0]) < 1*u.Angstrom) & (np.abs(sumer_overlap.spectral_axis[-1] - nnl_overlap.spectral_axis[-1]) < 1*u.Angstrom)
    except ValueError:
        assert False


def test_static_resample():
    try:
        sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
        nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high", emissions={"Source: I made it up": [1212, 1220],
                                                                                                    "Lyman-alpha":[1214-1218]})
        sumer_overlap, nnl_overlap = SolarSpectrum.spectral_overlap(sumer, nnl)
        resampled_sumer = SolarSpectrum.resample(sumer_overlap, new_axis = nnl_overlap.spectral_axis)
        assert(len(resampled_sumer.spectral_axis) == len(nnl_overlap.spectral_axis))
    except ValueError:
        assert False

def test_resample():
    try:
        sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
        nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high", emissions={"Source: I made it up": [1212, 1220],
                                                                                                    "Lyman-alpha":[1214-1218]})
        sumer_overlap, nnl_overlap = SolarSpectrum.spectral_overlap(sumer, nnl)
        sumer_overlap.resample(new_axis = nnl_overlap.spectral_axis)
        assert(len(sumer_overlap.spectral_axis) == len(nnl_overlap.spectral_axis))
    except ValueError:
        assert False

def test_static_convolve():
    
    try:
        sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
        sumer_convolved = SolarSpectrum.convolution(sumer, std=.3)
        assert not np.allclose(sumer.flux, sumer_convolved.flux)
    except ValueError:
        assert False


def test_convolve():

    try:
        sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
        flux = np.copy(sumer.flux.value)
        sumer.convolution(std=.3)
        assert not np.allclose(flux*sumer.flux.unit, sumer.flux)
    except ValueError:
        assert False


def test_stitch():

    try:
        sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
        nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high", emissions={"Source: I made it up": [1212, 1220],
                                                                                                    "Lyman-alpha":[1214-1218]})
        combined = SolarSpectrum.stitch(sumer, nnl)
        assert(len(combined.spectral_axis) > len(sumer.spectral_axis))
    except ValueError:
        assert False


def test_add_emissions():
    try:
        sumer = SolarSpectrum.sumer_spectrum()
        emissions = {"Lyman-alpha": [1559.5, 1563.0], "1302 OI Triplet": [1301, 1307], 
            "1562 CI": [1559, 1563]}
        sumer.add_emissions(emissions)
        sumer.emissions
        assert True
    except ValueError:
        assert False



def test_feature_fit():
    try:
        nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high", emissions={"Source: I made it up": [1212, 1220],
                                                                                          "Lyman-alpha":[1214-1218]})
        feature = [1301, 1303.5]
        height, mean, std, pixel_std = nnl.feature_fit(feature, height=5e-5, mean=1302.2, std=.4)
        assert True
    except ValueError:
        assert False


def test_daily_fit_poly():
    try:
        sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
        nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high", emissions={"Source: I made it up": [1212, 1220],                                                                                      "Lyman-alpha":[1214-1218]})
        feature = [1301, 1303.5]
        height, mean, std, pixel_std = nnl.feature_fit(feature, height=5e-5, mean=1302.2, std=.4)
        output, downsampled, dc, fit_results = SolarSpectrum.daily_fit(poly_degree=5, 
                                                             sumer=sumer,
                                                             daily_spec=nnl, 
                                                             gaussian_std = pixel_std,
                                                             fit='polynomial')
        assert True
    except ValueError:
        assert False


def test_daily_fit_legendre():
    try:
        sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
        nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high", emissions={"Source: I made it up": [1212, 1220],                                                                                      "Lyman-alpha":[1214-1218]})
        feature = [1301, 1303.5]
        height, mean, std, pixel_std = nnl.feature_fit(feature, height=5e-5, mean=1302.2, std=.4)
        output, downsampled, dc, fit_results = SolarSpectrum.daily_fit(poly_degree=5, 
                                                             sumer=sumer,
                                                             daily_spec=nnl, 
                                                             gaussian_std = pixel_std,
                                                             fit='legendre')
        assert True
    except ValueError:
        assert False


    