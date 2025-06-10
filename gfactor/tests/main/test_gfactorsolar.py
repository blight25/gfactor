import unittest
from datetime import date, timedelta
from astropy import units as u

import time
from datetime import date, timedelta

import shutil

from tqdm import tqdm

import math

from pathlib import Path

from gfactor.querying.LISIRDQuerying import LISIRDRetriever

import numpy as np

# Local imports
from gfactor.main.gfactorsolar import SolarSpectrum

class TestSolar(unittest.TestCase):

    NUM_SAMPLES = 5

    # Get the absolute path to the current file (LISIRDQuerying.py)
    current_file = Path(__file__).resolve()

    # Get the package root (assuming this file is always in gfactor/querying/)
    package_root = current_file.parents[2]  # gfactor/

    # Build path to target directory
    TEST_DIR = (package_root / "tests" / "main" / "spectra").as_posix()

    retriever = LISIRDRetriever()
    identifiers = retriever.irradiance_identifiers

    def test_load_sumer(self):
        try:
            sumer = SolarSpectrum.sumer_spectrum()
            self.assertIsInstance(sumer, SolarSpectrum)
        except ValueError:
            self.fail()


    def test_load_daily_new(self):

        print("Load Daily Spectrum (with Querying) Test")
        print(f"Number of samples per dataset: {TestSolar.NUM_SAMPLES}")

        for identifier in TestSolar.identifiers:

            # Date initialization - these will always be replaced
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            for subset in TestSolar.identifiers[identifier]:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, TestSolar.retriever.irradiance_datasets[dataset]['min_date'])
                max_date = min(max_date, TestSolar.retriever.irradiance_datasets[dataset]['max_date'])
            
            print(f"\nDataset {identifier}: minimum date of {min_date}, maximum date of {max_date}")

            total_days = (max_date - min_date).days
            interval = math.floor(total_days / TestSolar.NUM_SAMPLES)
            progress_bar = tqdm(total=total_days, desc=identifier)
            query_date = min_date
            while query_date <= max_date:
                dataframes = SolarSpectrum.daily_spectrum(date=query_date.strftime("%Y-%m-%d"), 
                                                  dataset=identifier)
                for dataframe in dataframes:
                    if dataframe:
                        self.assertIsInstance(dataframe, SolarSpectrum)
                query_date += timedelta(interval)
                progress_bar.update(interval)

    

    def test_load_daily(self):

        # Date initialization - these will always be replaced
        min_date = date(year=1600, month=1, day=1)
        max_date = date(year=2100, month=1, day=1)

        for identifier in TestSolar.identifiers:

            # Date initialization - these will always be replaced
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            for subset in TestSolar.identifiers[identifier]:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, TestSolar.retriever.irradiance_datasets[dataset]['min_date'])
                max_date = min(max_date, TestSolar.retriever.irradiance_datasets[dataset]['max_date'])

            total_days = (max_date - min_date).days
            interval = math.floor(total_days / TestSolar.NUM_SAMPLES)

            TestSolar.retriever.extract(dataset=identifier, 
                                        subset=None, 
                                        start_date=min_date.strftime("%Y-%m-%d"), 
                                        end_date=max_date.strftime("%Y-%m-%d"), 
                                        interval=interval,
                                        save_dir=TestSolar.TEST_DIR)
        
            query_date = min_date
            while query_date <= max_date:
                start_time = time.time()
                dataframes = SolarSpectrum.daily_spectrum(date=query_date.strftime("%Y-%m-%d"), dataset=identifier,
                                                          emissions=None, daily_dir=TestSolar.TEST_DIR)
                end_time = time.time()
                elapsed = (end_time - start_time)
                self.assertLess(elapsed, .1) # For pulling from pre-loaded files, shouldn't take more than a few milliseconds
                for dataframe in dataframes:
                    if dataframe:
                        self.assertIsInstance(dataframe, SolarSpectrum)
                query_date += timedelta(interval)
            
        # Remove test directory and associated files
        test_dir = Path(TestSolar.TEST_DIR)
        if test_dir.exists() and test_dir.is_dir():
            shutil.rmtree(Path(self.TEST_DIR))



    def test_spectral_overlap(self):
        sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
        nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high")
        sumer_overlap, nnl_overlap = SolarSpectrum.spectral_overlap(sumer, nnl)
        clipped_left = (np.abs(sumer_overlap.spectral_axis[0] - nnl_overlap.spectral_axis[0]) < 1*u.Angstrom)
        clipped_right = (np.abs(sumer_overlap.spectral_axis[-1] - nnl_overlap.spectral_axis[-1]) < 1*u.Angstrom)
        self.assertTrue(clipped_left & clipped_right)


    def test_static_resample(self):
        sumer = SolarSpectrum.sumer_spectrum(emissions={"Lyman-alpha":[1214, 1218]})
        nnl = SolarSpectrum.daily_spectrum(date="2020-09-15", dataset="NNL", res="high", emissions={"Test": [1212, 1220],
                                                                                                    "Lyman-alpha":[1214-1218]})
        sumer_overlap, nnl_overlap = SolarSpectrum.spectral_overlap(sumer, nnl)
        resampled_sumer = SolarSpectrum.resample(sumer_overlap, new_axis = nnl_overlap.spectral_axis)
        self.assertEqual(len(resampled_sumer.spectral_axis), len(nnl_overlap.spectral_axis))

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