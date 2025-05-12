import unittest
from datetime import datetime as dt
from datetime import date, timedelta
from astropy import units as u
from astropy.units import Quantity
from astropy.nddata import VarianceUncertainty, InverseVariance

import numpy as np

# Local imports
from gfactor.main.gfactorsolar import SolarSpectrum

class TestSolar(unittest.TestCase):

    def test_object(self):
        try:
            spectrum_object = SolarSpectrum(emissions={"Lyman-alpha": [1214, 1218], "Source: I made it up": [1212, 1220]}, 
                                            flux=[10, 10, 15, 10]*(u.W/u.m**2/u.Angstrom), 
                                            spectral_axis=[1210, 1214, 1218, 1222]*u.Angstrom,
                                            uncertainty=VarianceUncertainty([2, 2, 2, 2]*u.W**2/u.m**4/u.Angstrom**2))
            self.assertIsInstance(spectrum_object.emissions, dict)
        except ValueError:
            self.fail()


    def test_load_sumer(self):
        try:
            sumer = SolarSpectrum.sumer_spectrum()
            assert isinstance(sumer, SolarSpectrum)
        except ValueError:
            self.fail()

    def test_load_daily(self):
        try:
            nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high")
            assert isinstance(nnl, SolarSpectrum)
        except ValueError:
            self.fail()


    def test_spectral_overlap(self):
        try:
            sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
            nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high")
            sumer_overlap, nnl_overlap = SolarSpectrum.spectral_overlap(sumer, nnl)
            assert (np.abs(sumer_overlap.spectral_axis[0] - nnl_overlap.spectral_axis[0]) < 1*u.Angstrom) & (np.abs(sumer_overlap.spectral_axis[-1] - nnl_overlap.spectral_axis[-1]) < 1*u.Angstrom)
        except ValueError:
            self.fail()


    def test_static_resample(self):
        try:
            sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
            nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high", emissions={"Source: I made it up": [1212, 1220],
                                                                                                        "Lyman-alpha":[1214-1218]})
            sumer_overlap, nnl_overlap = SolarSpectrum.spectral_overlap(sumer, nnl)
            resampled_sumer = SolarSpectrum.resample(sumer_overlap, new_axis = nnl_overlap.spectral_axis)
            self.assertEqual(len(resampled_sumer.spectral_axis), len(nnl_overlap.spectral_axis))
        except ValueError:
            self.fail()

    def test_resample(self):
        try:
            sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
            nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high", emissions={"Source: I made it up": [1212, 1220],
                                                                                                        "Lyman-alpha":[1214-1218]})
            sumer_overlap, nnl_overlap = SolarSpectrum.spectral_overlap(sumer, nnl)
            sumer_overlap.resample(new_axis = nnl_overlap.spectral_axis)
            self.assertEqual(len(sumer_overlap.spectral_axis), len(nnl_overlap.spectral_axis))
        except ValueError:
            self.fail()

    def test_static_convolve(self):
        
        try:
            sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
            sumer_convolved = SolarSpectrum.convolution(sumer, std=.3)
            self.assertTrue(np.allclose(sumer.flux, sumer_convolved.flux))
        except ValueError:
            self.fail()


    def test_convolve(self):

        try:
            sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
            flux = np.copy(sumer.flux.value)
            sumer.convolution(std=.3)
            self.assertTrue(np.allclose(flux*sumer.flux.unit, sumer.flux))
        except ValueError:
            self.fail()


    def test_stitch(self):

        try:
            sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
            nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high", emissions={"Source: I made it up": [1212, 1220],
                                                                                                        "Lyman-alpha":[1214-1218]})
            combined = SolarSpectrum.stitch(sumer, nnl)
            self.assertGreater(len(combined.spectral_axis), len(sumer.spectral_axis))
        except ValueError:
            self.fail()


    def test_add_emissions(self):
        try:
            sumer = SolarSpectrum.sumer_spectrum()
            emissions = {"Lyman-alpha": [1559.5, 1563.0], "1302 OI Triplet": [1301, 1307], 
                "1562 CI": [1559, 1563]}
            sumer.add_emissions(emissions)
            sumer.emissions
            self.assertTrue(True)
        except ValueError:
            self.fail()



    def test_feature_fit(self):
        try:
            nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high", emissions={"Source: I made it up": [1212, 1220],
                                                                                            "Lyman-alpha":[1214-1218]})
            feature = [1301, 1303.5]
            height, mean, std, pixel_std = nnl.feature_fit(feature, height=5e-5, mean=1302.2, std=.4)
            assert True
        except ValueError:
            self.fail()


    def test_daily_fit_poly(self):
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
            self.fail()


    def test_daily_fit_legendre(self):
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
            self.fail()


    