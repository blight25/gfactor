# Standard library
import unittest
import json

from gfactor.querying.LISIRDQuerying import LISIRDRetriever, NoDataAvailableError
from gfactor.querying.NISTQuerying import NISTRetriever

import shutil
from pathlib import Path

import math
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

from datetime import date, timedelta


class TestLISIRDQuerying(unittest.TestCase):

    NUM_SAMPLES = 30 # Total samples to fetch per dataset
    FAILED_QUERY_TOLERANCE = 3 # Failed dates
    TIMEOUT = 20 # seconds waiting for response
    TEST_DIR = "./gfactor/tests/spectra"
    retriever = LISIRDRetriever()
    irradiance_identifiers = retriever.irradiance_identifiers
    other_identifiers = retriever.other_identifiers
    STATUS_LOG = "./gfactor/tests/querying/spectral_test_log.json"


    def _get_working_identifiers(self):
            """
            Helper to return only identifiers and subsets with status 'working' in the status log.
            Returns a dict: {identifier: [subsets]}
            """
            if Path(self.STATUS_LOG).exists():
                with open(self.STATUS_LOG, "r") as f:
                    status_log = json.load(f)
            else:
                return {**self.irradiance_identifiers, **self.other_identifiers}

            working = {}
            for identifier, value in status_log.items():
                if isinstance(value, dict) and all(isinstance(v, dict) for v in value.values()):
                    # Has subsets
                    good_subsets = [subset for subset, subval in value.items() if subval.get("status") == "working"]
                    if good_subsets:
                        working[identifier] = good_subsets
                else:
                    if value.get("status") == "working":
                        working[identifier] = [None]
            return working
    

    def test_retrieve_legal(self):

        required_cols = ["wavelength (nm)", "irradiance (W/m^2/nm)"]

        print("\nIrradiance Querying Test")
        print(f"Number of samples per dataset: {TestLISIRDQuerying.NUM_SAMPLES}")
 
        identifiers = {**TestLISIRDQuerying.retriever.irradiance_identifiers, **TestLISIRDQuerying.retriever.other_identifiers}
        
        # Load or initialize status log
        if Path(self.STATUS_LOG).exists():
            with open(self.STATUS_LOG, "r") as f:
                status_log = json.load(f)
        else:
            status_log = {}
        
        # Mark all as working initially
        for identifier in identifiers:
            subsets = identifiers[identifier]
            if subsets[-1]:
                status_log[identifier] = {subset: {"status": "working", "bad dates": []} for subset in subsets}
            else:
                status_log[identifier] = {"status": "working", "bad dates": []}
        with open(self.STATUS_LOG, "w") as f:
            json.dump(status_log, f, indent=2)

        # Dataset category (affects dataframe type checking)
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
            
            print(f"\nDataset {identifier}: minimum date of {min_date}, maximum date of {max_date}")

            total_days = (max_date - min_date).days
            interval = math.floor(total_days / TestLISIRDQuerying.NUM_SAMPLES)
            progress_bar = tqdm(total=total_days, desc=identifier)
            query_date = min_date
            
            failed_queries = 0
            timeout = np.zeros_like(subsets, dtype=bool) # Boolean array indicating which subsets, if any, have timed out
            while query_date < max_date:
                # Query
                for i, subset in enumerate(subsets):
                    if not timeout[i]:
                        try:
                            df = TestLISIRDQuerying.retriever.retrieve(
                                dataset=identifier,
                                subset=subset,
                                query_date=query_date.strftime("%Y-%m-%d"),
                                timeout=self.TIMEOUT
                            )
                        except RuntimeError:
                            if subset is not None:
                                print(f"\nWARNING: Dataset '{identifier + "_" + subset}' timed out after {self.TIMEOUT} seconds - it may no longer be available through the LISIRD API.\n")
                                status_log[identifier][subset]["status"] = "timeout"
                                status_log[identifier][subset]["timeout"] = self.TIMEOUT

                            else:
                                print(f"\nWARNING: Dataset '{identifier}' timed out after {self.TIMEOUT} seconds - it may no longer be available through the LISIRD API.\n")
                                status_log[identifier]["status"] = "timeout"
                                status_log[identifier]["timeout"] = self.TIMEOUT
                            
                            # Break out of both loops to advance to next dataset
                            timeout[i] = True
                            continue
                        except NoDataAvailableError:
                            if subset is not None:
                                status_log[identifier][subset]["bad dates"].append(query_date.strftime("%Y-%m-%d"))
                            else:
                                status_log[identifier]["bad dates"].append(query_date.strftime("%Y-%m-%d"))
                            failed_queries += 1
                        
                        # For available dataframes, ensure data is as expected
                        self.assertIsNotNone(df)
                        self.assertGreater(len(df.values), 0)
                        if extended_check:
                            for col in required_cols:
                                self.assertIn(col, df.columns)
                                dtype = df[col].dtype
                                python_dtype = float if np.issubdtype(dtype, np.floating) else None
                                self.assertEqual(python_dtype, float)
                
                # Move on to next dataset if timeout occured
                if np.all(timeout):
                    break
  
                # Only increment if we didn't break due to timeout
                query_date += timedelta(interval)
                progress_bar.update(interval)
                continue
            
            progress_bar.close()

            with open(self.STATUS_LOG, "w") as f:
                json.dump(status_log, f, indent=2)
            self.assertLess(failed_queries, TestLISIRDQuerying.FAILED_QUERY_TOLERANCE)
    

    def test_dataset_validity(self):
        """
        Checks that all datasets in the status log have a 'working' status. Fails if any dataset or subset is not 'working'.
        """
        if Path(self.STATUS_LOG).exists():
            with open(self.STATUS_LOG, "r") as f:
                status_log = json.load(f)
        else:
            self.fail(f"Status log file {self.STATUS_LOG} does not exist.")

        for dataset, value in status_log.items():
            # If value is a dict of subsets
            if isinstance(value, dict) and all(isinstance(v, dict) for v in value.values()):
                for subset, subval in value.items():
                    status = subval.get("status", None)
                    self.assertEqual(status, "working", f"Dataset '{dataset}', subset '{subset}' has status '{status}' (expected 'working')")
            else:
                status = value.get("status", None)
                self.assertEqual(status, "working", f"Dataset '{dataset}' has status '{status}' (expected 'working')")  


    def test_retrieve_invalid_dataset(self):

        illegal_types = [None, 5, 100.7, False]
        illegal_dataset = 'test'

        for illegal_type in illegal_types:
            with self.assertRaises(TypeError):
                self.retriever.retrieve(dataset=illegal_type, query_date="2012-01-01", subset=None, timeout=self.TIMEOUT)
        
        with self.assertRaises(ValueError):
            self.retriever.retrieve(dataset=illegal_dataset, query_date="2012-01-01", subset=None, timeout=self.TIMEOUT)

    
    def test_retrieve_invalid_subset(self):

        illegal_types = [5, 100.7, False]
        illegal_subset = 'test'

        identifiers = self._get_working_identifiers()
        
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
            
            # Variables
            query_date = min_date.strftime("%Y-%m-%d")
            subset = subsets[-1]

            # Subset exists, failure to specify should throw error
            if subset is not None:
                with self.assertRaises(ValueError):
                    self.retriever.retrieve(dataset=identifier, query_date=query_date, subset=None, timeout=self.TIMEOUT)
            
            # No subset exists, attempting to specify should throw error
            else:
                with self.assertRaises(ValueError):
                    self.retriever.retrieve(dataset=identifier, query_date=query_date, subset=illegal_subset, timeout=self.TIMEOUT)
            
            # Invalid typing
            for illegal_type in illegal_types:
                with self.assertRaises(TypeError):
                    self.retriever.retrieve(dataset=identifier, query_date=query_date, subset=illegal_type, timeout=self.TIMEOUT)
    

    def test_retrieve_invalid_date(self):

        illegal_types = [None, 5, 100.7, False]
        
        identifiers = self._get_working_identifiers()
        
        for identifier in identifiers:

            if identifier in TestLISIRDQuerying.irradiance_identifiers:
                datasets = TestLISIRDQuerying.retriever.irradiance_datasets
        
            elif identifier in TestLISIRDQuerying.other_identifiers:
                datasets = TestLISIRDQuerying.retriever.other_datasets
            
            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            # Only take the date range that works for all subsets
            subsets = identifiers[identifier]
            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            # Variables
            early_date = min_date - timedelta(days=10)
            early_date = early_date.strftime("%Y-%m-%d")
            late_date = max_date + timedelta(days=10)
            late_date = late_date.strftime("%Y-%m-%d")

            # Below minimum and above maximum
            for query_date in (early_date, late_date):
                with self.assertRaises(ValueError):
                    self.retriever.retrieve(dataset=identifier, query_date=query_date, subset=subset, timeout=self.TIMEOUT)

            # Invalid typing
            for illegal_type in illegal_types:
                with self.assertRaises(TypeError):
                    self.retriever.retrieve(dataset=identifier, query_date=illegal_type, subset=subset, timeout=self.TIMEOUT)
    

    def test_extract_legal(self):
        
        # For irradiance datatsets only
        required_cols = ["wavelength (nm)", "irradiance (W/m^2/nm)"]

        # Dataset names and subsets
        identifiers = self._get_working_identifiers()
        all_identifiers = {**self.irradiance_identifiers, **self.other_identifiers}

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

            # Only take the date range that works for all subsets
            subsets = identifiers[identifier]

            # Ensure that this dataset and ALL of its subsets (if they exist) are operational before testing
            if identifier not in all_identifiers or not len(subsets) == len(all_identifiers[identifier]):
                continue

            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            # Variables
            interval = (max_date - min_date).days - 1
            start_date = min_date.strftime("%Y-%m-%d")
            end_date = min_date + timedelta(days=interval)
            end_date = end_date.strftime("%Y-%m-%d")

            # Extract
            TestLISIRDQuerying.retriever.extract(
                dataset=identifier,
                subset=None,
                interval=interval,
                save_dir=TestLISIRDQuerying.TEST_DIR,
                start_date=start_date,
                end_date=end_date,
                overwrite=True,
                timeout=self.TIMEOUT
            )
            
            # Check that files exist
            local_dir = Path(TestLISIRDQuerying.TEST_DIR + "/" + identifier)
            local_dir.mkdir(parents=True, exist_ok=True)

            # Subsets
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

        identifiers = self._get_working_identifiers()

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
                
                # Variables
                interval = (max_date - min_date).days - 1
                start_date = min_date.strftime("%Y-%m-%d")
                end_date = min_date + timedelta(days=interval)
                end_date = end_date.strftime("%Y-%m-%d")

                # Use first subset for testing
                TestLISIRDQuerying.retriever.extract(
                    dataset=identifier,
                    subset=subsets[0],
                    interval=interval,
                    save_dir=TestLISIRDQuerying.TEST_DIR,
                    start_date=start_date,
                    end_date=end_date,
                    overwrite=True,
                    timeout=self.TIMEOUT
                )
                
                # Check that files exist
                local_dir = Path(TestLISIRDQuerying.TEST_DIR + "/" + identifier)
                local_dir.mkdir(parents=True, exist_ok=True)

                # File testing
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
                self.retriever.extract(dataset=illegal_type, start_date=None,
                                      end_date=None, interval=1,
                                      save_dir=self.TEST_DIR,
                                      subset=None, timeout=self.TIMEOUT)

            # Remove test directory and associated files
            if test_dir.exists() and test_dir.is_dir():
                shutil.rmtree(Path(self.TEST_DIR))
            
        with self.assertRaises(ValueError):
            self.retriever.extract(dataset=illegal_dataset, start_date=None,
                                   end_date=None, interval=1,
                                   save_dir=self.TEST_DIR,
                                   subset=None, timeout=self.TIMEOUT)

            # Remove test directory and associated files
            if test_dir.exists() and test_dir.is_dir():
                shutil.rmtree(Path(self.TEST_DIR))
    

    def test_extract_invalid_subset(self):

        illegal_types = [5, 100.7, False]
        illegal_subset = 'test'

        identifiers = self._get_working_identifiers()
        
        for identifier in identifiers:

            if identifier in TestLISIRDQuerying.irradiance_identifiers:
                datasets = TestLISIRDQuerying.retriever.irradiance_datasets
        
            elif identifier in TestLISIRDQuerying.other_identifiers:
                datasets = TestLISIRDQuerying.retriever.other_datasets
            
            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            # Only take the date range that works for all subsets
            subsets = identifiers[identifier]
            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            # Type Checking 
            for illegal_type in illegal_types:
                with self.assertRaises(TypeError):
                    self.retriever.extract(dataset=identifier, subset=illegal_type,
                                          interval=1, save_dir=self.TEST_DIR,
                                          start_date=min_date,
                                          end_date=max_date,
                                          overwrite=True,
                                          timeout=self.TIMEOUT)
            
            # Illegal Value
            subset = subsets[-1]        
            if subset is not None:
                with self.assertRaises(ValueError):
                    self.retriever.extract(dataset=identifier, subset=illegal_subset,
                                          interval=1, save_dir=self.TEST_DIR,
                                          start_date=min_date,
                                          end_date=max_date,
                                          overwrite=True,
                                          timeout=self.TIMEOUT)
                    

    def test_extract_invalid_date(self):

        illegal_types = [None, 5, 100.7, False]
        
        identifiers = self._get_working_identifiers()

        test_dir = Path(TestLISIRDQuerying.TEST_DIR)
        
        for identifier in identifiers:

            if identifier in TestLISIRDQuerying.irradiance_identifiers:
                datasets = TestLISIRDQuerying.retriever.irradiance_datasets
        
            elif identifier in TestLISIRDQuerying.other_identifiers:
                datasets = TestLISIRDQuerying.retriever.other_datasets
            
            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            # Only take the date range that works for all subsets
            subsets = identifiers[identifier]
            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            # Variables
            early_date = min_date - timedelta(days=10)
            early_date = early_date.strftime("%Y-%m-%d")
            late_date = max_date + timedelta(days=10)
            late_date = late_date.strftime("%Y-%m-%d")

            # Below minimum and above maximum
            for begin_date in (None, early_date):
                for finish_date in (None, late_date):
                    if begin_date is None and finish_date is None:
                        continue
                    with self.assertRaises(ValueError):
                        self.retriever.extract(dataset=identifier, start_date=begin_date,
                                              end_date=finish_date, interval=1,
                                              save_dir=self.TEST_DIR,
                                              subset=None, timeout=self.TIMEOUT)

            # Type checking
            for i in range(len(illegal_types)):
                for j in range(len(illegal_types)):
                    if illegal_types[i] is None and illegal_types[j] is None:
                        continue
                    with self.assertRaises(TypeError):
                        self.retriever.extract(dataset=identifier, start_date=illegal_types[i],
                                              end_date=illegal_types[j], interval=1,
                                              save_dir=self.TEST_DIR,
                                              subset=None, timeout=self.TIMEOUT)
            
        # Remove test directory and associated files
        if test_dir.exists() and test_dir.is_dir():
            shutil.rmtree(Path(self.TEST_DIR))


class TestNISTQuerying(unittest.TestCase):

    NUM_SAMPLES = 10 # Atomic samples in extract
    TEST_DIR = "./gfactor/tests/spectra"
    required_cols = {"obs_wl(A)" : float, 
                    "fik" : float, 
                    "term_k" : str,  
                    "conf_k": str, 
                    "Ek(eV)": float,
                    "J_k": float,
                    "term_i": str, 
                    "conf_i": str, 
                    "Ei(eV)": float,
                    "J_i": float,
                    "Acc" : float, 
                    "Aki(s^-1)" : float, 
                    "element" : str, 
                    "sp_num": float}
    
    optional_cols = {"unc_obs_wl": float}

    elements = [
                'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ar', 'Ca', 
                'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Co', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 
                'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'I', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 
                'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 
                'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Pa', 'Th', 'Np', 'U', 'Am', 'Pu', 'Cm', 
                'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Bh', 'Sg', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 
                'Lv', 'Ts', 'Og'
                ]

    # Known to fail: 'U' yields ValueError, all other queries proceed but produce invalid DataFrames
    problem_elements = {'Mt', 'At', 'Fm', 'U', 'Rf', 'Cf', 'Pa', 'No', 'Sg', 'Am', 'Pr', 'Nb', 
                        'Lr', 'Rn', 'Lv', 'Zr', 'Os', 'Hs', 'Og', 'Mc', 'Cm', 'Fl', 'Cn', 'Db', 
                        'Bk', 'Re', 'Rg', 'Po', 'Tb', 'Ds', 'Es', 'Th', 'Pm', 'Se', 'Bh', 'Nh', 
                        'Ts', 'Md', 'Pu', 'Np'}

    retriever = NISTRetriever()


    def _map_numpy_dtype_to_python(self, dtype):
        """
        Map a numpy/pandas dtype to the corresponding Python type (float, int, str, or None).
        Handles pandas extension types like string[python].
        """

        if pd.api.types.is_string_dtype(dtype):
            return str
        elif np.issubdtype(dtype, np.floating):
            return float
        elif np.issubdtype(dtype, np.integer):
            return int
        else:
            return None


    def test_retrieve_legal(self):

        # Loop through random subset of elements (full query is infeasible for reasonable runtimes)
        elements = random.sample(TestNISTQuerying.elements, k=TestNISTQuerying.NUM_SAMPLES)

        for el in elements:
            if el in TestNISTQuerying.problem_elements:
                    results = TestNISTQuerying.retriever.retrieve(elements=[el], ionized=False)
                    df = results[el]
                    self.assertIsNone(df) # Elements with bad data are set to None
            else:
                for ionization in (True, False):
                    results = TestNISTQuerying.retriever.retrieve(elements=[el], ionized=ionization)
                    df = results[el]
                    for col in TestNISTQuerying.required_cols:
                        self.assertIn(col, df.columns)
                        dtype = df[col].dtype
                        python_type = self._map_numpy_dtype_to_python(dtype)
                        self.assertIsNotNone(python_type)
                        self.assertEqual(python_type, TestNISTQuerying.required_cols[col])
                    for col in TestNISTQuerying.optional_cols:
                        if col in df.columns:
                            dtype = df[col].dtype
                            python_type = self._map_numpy_dtype_to_python(dtype)
                            self.assertIsNotNone(python_type)
                            self.assertEqual(python_type, TestNISTQuerying.optional_cols[col])
    

    def test_retrieve_invalid_elements(self):

        illegal_types = [None, 5, 100.7, False]
        illegal_element = "test"

        for type in illegal_types:
            with self.assertRaises(TypeError):
                results = TestNISTQuerying.retriever.retrieve(elements=[type], ionized=False)
        
        with self.assertRaises(ValueError):
            results = TestNISTQuerying.retriever.retrieve(elements=[illegal_element], ionized=False)
    
    
    def test_retrieve_invalid_ionization(self):

        illegal_types = [None, 5, 100.7, "test"]
        for type in illegal_types:
            with self.assertRaises(TypeError):
                results = TestNISTQuerying.retriever.retrieve(elements=["H"], ionized=type)