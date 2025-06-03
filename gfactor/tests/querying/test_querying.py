# Standard library
import unittest

from gfactor.querying.LISIRDQuerying import LISIRDRetriever, NoDataAvailableError
from gfactor.querying.NISTQuerying import NISTRetriever

import shutil
from pathlib import Path

import math

import numpy as np
import pandas as pd

import requests
import time

from tqdm import tqdm

from datetime import date, timedelta


class TestLISIRDQuerying(unittest.TestCase):

    NUM_SAMPLES = 30
    FAILED_QUERY_TOLERANCE = 3
    TEST_DIR = "./gfactor/tests/spectra"
    retriever = LISIRDRetriever()
    irradiance_identifiers = retriever.irradiance_identifiers
    other_identifiers = retriever.other_identifiers
    error_loc = "data/errors"


    def test_retrieve_legal(self):

        required_cols = ["wavelength (nm)", "irradiance (W/m^2/nm)"]

        print("\nIrradiance Querying Test")
        print(f"Number of samples per dataset: {TestLISIRDQuerying.NUM_SAMPLES}")
 
        identifiers = {**TestLISIRDQuerying.retriever.irradiance_identifiers, **TestLISIRDQuerying.other_identifiers}
        
        for identifier in identifiers:
            
            extended_check = False
            if identifier in TestLISIRDQuerying.irradiance_identifiers:
                datasets = TestLISIRDQuerying.retriever.irradiance_datasets
                extended_check = True

            elif identifier in TestLISIRDQuerying.other_identifiers:
                datasets = TestLISIRDQuerying.retriever.other_datasets


            # Create error file for recording problematic dates
            error_dir = Path(TestLISIRDQuerying.error_loc + "/" + identifier)
            error_dir.mkdir(parents=True, exist_ok=True)
            error_file = error_dir / "problem_dates_test.txt"
            failed_queries = 0
            with open (error_file, "w") as problem_file: 
                problem_file.write("UNSUCCESSFUL QUERY DATES\n\n")
            print(f"\nError log to be saved in directory {error_dir}")

            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            subsets = identifiers[identifier]

            # Only take the date range that works for all subsets
            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            print(f"\nDataset {identifier}: minimum date of {min_date}, maximum date of {max_date}")

            total_days = (max_date - min_date).days
            interval = math.floor(total_days / TestLISIRDQuerying.NUM_SAMPLES)
            progress_bar = tqdm(total=total_days, desc=identifier)
            query_date = min_date
            
            while query_date <= max_date:
                
                # Query
                for subset in subsets:
                    try:
                        df = TestLISIRDQuerying.retriever.retrieve(dataset=identifier, subset=subset,
                                                                   query_date = query_date.strftime("%Y-%m-%d"))
                           
                    except requests.exceptions.RequestException: # sometimes SSL behaves weird, or the connection gets closed suddenly/times out: try one more time
                        time.sleep(5) # First, give server some time to breath
                        df = TestLISIRDQuerying.retriever.retrieve(dataset=identifier, subset=subset,
                                                                   query_date = query_date.strftime("%Y-%m-%d"))
                    
                    except NoDataAvailableError: # There was no data available
                        with open (error_file, "a") as problem_file:
                            problem_file.write(f"\n{query_date}\n") # Keep tabs on problematic dates
                        failed_queries += 1
                        query_date += timedelta(interval)
                        progress_bar.update(1)
                        continue
                        
                    # For available dataframes, ensure data is as expected
                    self.assertIsNotNone(df) # Dataframe exists
                    self.assertGreater(len(df.values), 0) # Contains actual data

                    if extended_check:
                        for col in required_cols:
                            self.assertIn(col, df.columns) # required columns are present
                            dtype = df[col].dtype
                            python_dtype = float if np.issubdtype(dtype, np.floating) else None
                            self.assertEqual(python_dtype, float) # All dtypes are float

                # Increment
                query_date += timedelta(interval)
                progress_bar.update(interval)

            print(f"\nNumber of failed queries: {failed_queries}")

            self.assertLess(failed_queries, TestLISIRDQuerying.FAILED_QUERY_TOLERANCE)


    def test_retrieve_default_subset(self):

        identifiers = {**TestLISIRDQuerying.retriever.irradiance_identifiers, **TestLISIRDQuerying.other_identifiers}
        
        for identifier in identifiers:

            if identifier in TestLISIRDQuerying.irradiance_identifiers:
                datasets = TestLISIRDQuerying.retriever.irradiance_datasets
        
            elif identifier in TestLISIRDQuerying.other_identifiers:
                datasets = TestLISIRDQuerying.retriever.other_datasets
            
            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            subsets = identifiers[identifier]

            # Only take the date range that works for all subsets
            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            query_date = min_date.strftime("%Y-%m-%d")

            # Test without specifying subset
            TestLISIRDQuerying.retriever.retrieve(dataset=identifier, query_date=query_date, subset=None)
        
        self.assertTrue(True)    


    def test_retrieve_invalid_dataset(self):

        illegal_types = [None, 5, 100.7, False]
        illegal_dataset = 'test'

        for illegal_type in illegal_types:
            with self.assertRaises(TypeError):
                TestLISIRDQuerying.retriever.retrieve(dataset=illegal_type, query_date="2012-01-01", subset=None)
        
        with self.assertRaises(ValueError):
            TestLISIRDQuerying.retriever.retrieve(dataset=illegal_dataset, query_date="2012-01-01", subset=None)

        self.assertTrue(True)

    
    def test_retrieve_invalid_subset(self):

        illegal_types = [5, 100.7, False]
        illegal_subset = 'test'

        identifiers = {**TestLISIRDQuerying.irradiance_identifiers, **TestLISIRDQuerying.other_identifiers}
        
        for identifier in identifiers:

            if identifier in TestLISIRDQuerying.irradiance_identifiers:
                datasets = TestLISIRDQuerying.retriever.irradiance_datasets
        
            elif identifier in TestLISIRDQuerying.other_identifiers:
                datasets = TestLISIRDQuerying.retriever.other_datasets
            
            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            subsets = identifiers[identifier]

            # Only take the date range that works for all subsets
            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            query_date = min_date.strftime("%Y-%m-%d")
            subset = subsets[-1]

            # Invalid subset
            for illegal_type in illegal_types:
                with self.assertRaises(TypeError):
                    TestLISIRDQuerying.retriever.retrieve(dataset=identifier, query_date=query_date, subset=illegal_type)
            if subset is not None:
                with self.assertRaises(ValueError):
                    TestLISIRDQuerying.retriever.retrieve(dataset=identifier, query_date=query_date, subset=illegal_subset)
    

    def test_retrieve_invalid_date(self):

        illegal_types = [None, 5, 100.7, False]
        
        identifiers = {**TestLISIRDQuerying.irradiance_identifiers, **TestLISIRDQuerying.other_identifiers}
        
        for identifier in identifiers:

            if identifier in TestLISIRDQuerying.irradiance_identifiers:
                datasets = TestLISIRDQuerying.retriever.irradiance_datasets
        
            elif identifier in TestLISIRDQuerying.other_identifiers:
                datasets = TestLISIRDQuerying.retriever.other_datasets
            
            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            subsets = identifiers[identifier]

            # Only take the date range that works for all subsets
            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            early_date = min_date - timedelta(days=10)
            early_date = early_date.strftime("%Y-%m-%d")
            late_date = max_date + timedelta(days=10)
            late_date = late_date.strftime("%Y-%m-%d")

            # Type checking
            for illegal_type in illegal_types:
                with self.assertRaises(TypeError):
                    TestLISIRDQuerying.retriever.retrieve(dataset=identifier, query_date=illegal_type, subset=subset)

            # Below minimum and above maximum
            for query_date in (early_date, late_date):
                with self.assertRaises(ValueError):
                    TestLISIRDQuerying.retriever.retrieve(dataset=identifier, query_date=query_date, subset=subset)
        
        self.assertTrue(True)
    

    def test_extract_legal(self):

        required_cols = ["wavelength (nm)", "irradiance (W/m^2/nm)"]

        identifiers = {**TestLISIRDQuerying.irradiance_identifiers, **TestLISIRDQuerying.other_identifiers}

        test_dir = Path(TestLISIRDQuerying.TEST_DIR)

        for identifier in identifiers:
            
            extended_check = False
            if identifier in TestLISIRDQuerying.irradiance_identifiers:
                datasets = TestLISIRDQuerying.retriever.irradiance_datasets
                extended_check = True

            elif identifier in TestLISIRDQuerying.other_identifiers:
                datasets = TestLISIRDQuerying.retriever.other_datasets
            
            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            subsets = identifiers[identifier]

            # Only take the date range that works for all subsets
            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            interval = (max_date - min_date).days - 1
            start_date = min_date.strftime("%Y-%m-%d")
            end_date = min_date + timedelta(days=interval)
            end_date = end_date.strftime("%Y-%m-%d")
            TestLISIRDQuerying.retriever.extract(dataset=identifier, subset=None,
                                                 interval=interval, save_dir=TestLISIRDQuerying.TEST_DIR,
                                                 start_date=start_date,
                                                 end_date=end_date,
                                                 overwrite=True)
            
            # Check that files exist
            local_dir = Path(TestLISIRDQuerying.TEST_DIR + "/" + identifier)
            local_dir.mkdir(parents=True, exist_ok=True)

            for subset in subsets:
                for query_date in (start_date, end_date):
                    file = identifier + "_" + subset + ".pickle" if subset else identifier + ".pickle"
                    file = local_dir / query_date / file
                    df = pd.read_pickle(file)

                    # For available dataframes, ensure data is as expected
                    self.assertIsNotNone(df) # Dataframe exists
                    self.assertGreater(len(df.values), 0) # Contains actual data

                    if extended_check: # For irradiance datasets only
                        for col in required_cols:
                            self.assertIn(col, df.columns) # required columns are present
                            dtype = df[col].dtype
                            python_dtype = float if np.issubdtype(dtype, np.floating) else None
                            self.assertEqual(python_dtype, float) # All dtypes are float
            

        # Remove test directory and associated files
        if test_dir.exists() and test_dir.is_dir():
            shutil.rmtree(Path(TestLISIRDQuerying.TEST_DIR))
            

    def test_extract_specific_subset(self):

        required_cols = ["wavelength (nm)", "irradiance (W/m^2/nm)"]

        identifiers = {**TestLISIRDQuerying.irradiance_identifiers, **TestLISIRDQuerying.other_identifiers}

        test_dir = Path(TestLISIRDQuerying.TEST_DIR)

        for identifier in identifiers:

            subsets = identifiers[identifier]
            
            # Case where subsets = [None] is covered by the base test
            if len(subsets) > 1:
            
                extended_check = False
                if identifier in TestLISIRDQuerying.irradiance_identifiers:
                    datasets = TestLISIRDQuerying.retriever.irradiance_datasets
                    extended_check = True

                elif identifier in TestLISIRDQuerying.other_identifiers:
                    datasets = TestLISIRDQuerying.retriever.other_datasets
                
                # Date initialization - these will always be replaced once actual dates are located
                min_date = date(year=1600, month=1, day=1)
                max_date = date(year=2100, month=1, day=1)

                # Only take the date range that works for all subsets
                for subset in subsets:
                    dataset = identifier + "_" + subset if subset else identifier
                    min_date = max(min_date, datasets[dataset]['min_date'])
                    max_date = min(max_date, datasets[dataset]['max_date'])
                
                interval = (max_date - min_date).days - 1
                start_date = min_date.strftime("%Y-%m-%d")
                end_date = min_date + timedelta(days=interval)
                end_date = end_date.strftime("%Y-%m-%d")

                # Use first subset for testing
                TestLISIRDQuerying.retriever.extract(dataset=identifier, subset=subsets[0],
                                                    interval=interval, save_dir=TestLISIRDQuerying.TEST_DIR,
                                                    start_date=start_date,
                                                    end_date=end_date,
                                                    overwrite=True)
                
                # Check that files exist
                local_dir = Path(TestLISIRDQuerying.TEST_DIR + "/" + identifier)
                local_dir.mkdir(parents=True, exist_ok=True)
            
                for query_date in (start_date, end_date):

                    # File for first subset should exist and be usable
                    file = identifier + "_" + subsets[0] + ".pickle"
                    file = local_dir / query_date / file
                    df = pd.read_pickle(file)

                    # For available dataframes, ensure data is as expected
                    self.assertIsNotNone(df) # Dataframe exists
                    self.assertGreater(len(df.values), 0) # Contains actual data

                    if extended_check: # For irradiance datasets only
                        for col in required_cols:
                            self.assertIn(col, df.columns) # required columns are present
                            dtype = df[col].dtype
                            python_dtype = float if np.issubdtype(dtype, np.floating) else None
                            self.assertEqual(python_dtype, float) # All dtypes are float
                    
                    # File for second subset shouldn't exist
                    invalid_file = file = identifier + "_" + subsets[1] + ".pickle"
                    invalid_file = local_dir / query_date / invalid_file
                    with self.assertRaises(FileNotFoundError):
                        df = pd.read_pickle(file)
        
        # Remove test directory and associated files
        if test_dir.exists() and test_dir.is_dir():
            shutil.rmtree(Path(TestLISIRDQuerying.TEST_DIR))
    
    
    def test_extract_invalid_dataset(self):

        illegal_types = [None, 5, 100.7, False]
        illegal_dataset = 'test'
        test_dir = Path(TestLISIRDQuerying.TEST_DIR)

        for illegal_type in illegal_types:
            with self.assertRaises(TypeError):
                TestLISIRDQuerying.retriever.extract(dataset=illegal_type, start_date=None,
                                                     end_date=None, interval=1,
                                                     save_dir=TestLISIRDQuerying.TEST_DIR,
                                                     subset=None)

            # Remove test directory and associated files
            if test_dir.exists() and test_dir.is_dir():
                shutil.rmtree(Path(TestLISIRDQuerying.TEST_DIR))
            
        with self.assertRaises(ValueError):
            TestLISIRDQuerying.retriever.extract(dataset=illegal_dataset, start_date=None,
                                                     end_date=None, interval=1,
                                                     save_dir=TestLISIRDQuerying.TEST_DIR,
                                                     subset=None)

            # Remove test directory and associated files
            if test_dir.exists() and test_dir.is_dir():
                shutil.rmtree(Path(TestLISIRDQuerying.TEST_DIR))