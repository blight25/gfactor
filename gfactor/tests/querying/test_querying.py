# Standard library
import unittest
import datetime as dt
import random

# Local imports
from gfactor import NISTRetriever, LISIRDRetriever

class TestQuerying(unittest.TestCase):

    def test_NISTRetrieval(self):
        
        retriever = NISTRetriever()
        
        # Set random seed for reproducibility
        random_seed = 5
        random.seed(random_seed)
        
        # For checking dataframe validity
        core_cols = ['Ei(eV)', 'Ek(eV)', 'fik', 'J_i', 'J_k', 'Acc', 'conf_i', 'term_i', 'conf_k', 'term_k']
        wavelength_col = 'obs_wl(A)'
        
        # Single element, non-ionized:
        for element in retriever.elements:
            try:
                df = retriever.retrieve(wavelength_bounds=[5000, 9000], elements=[element], ionized=False)
                missing_cores = [col for col in core_cols if col not in df.columns]
                self.assertTrue(len(df) > 0 and len(missing_cores) == 0 and wavelength_col in df.columns)
            except ValueError:
                self.fail()
        
        # Single element, ionized:
        for element in retriever.elements:
            try:
                df = retriever.retrieve(wavelength_bounds=[5000, 9000], elements=[element], ionized=True)
                missing_cores = [col for col in core_cols if col not in df.columns]
                assert len(df) > 0 and len(missing_cores) == 0 and wavelength_col in df.columns   
            except ValueError:
                self.fail()
        
        # Multiple elements, non-ionized (can't check all combinations, so just randomly pick a few)
        atoms = random.choices(retriever.elements, k=random.randint(0, len(retriever.elements)))
        try:
            df = retriever.retrieve(wavelength_bounds=[5000, 9000], elements=atoms, ionized=False)
            missing_cores = [col for col in core_cols if col not in df.columns]
            assert len(df) > 0 and len(missing_cores) == 0 and wavelength_col in df.columns
        except ValueError:
            self.fail()
        
        # Multiple elements, ionized:
        atoms = random.choices(retriever.elements, k=random.randint(0, len(retriever.elements)))
        try:
            df = retriever.retrieve(wavelength_bounds=[5000, 9000], elements=atoms, ionized=True)
            missing_cores = [col for col in core_cols if col not in df.columns]
            assert len(df) > 0 and len(missing_cores) == 0 and wavelength_col in df.columns
        except ValueError:
            self.fail()



    def test_LISIRDQuerying(self):
        
        retriever = LISIRDRetriever()
        
        # Cover all datasets
        for key in list(retriever.datasets.keys()):
            
            # ************* Variables *************
            
            # Get minimum date
            min_date = retriever.datasets[key]['min_date']
            
            # Get maximum date
            max_date = retriever.datasets[key]['max_date']
            
            # Dates outside bounds (for error testing)
            below_min_date = min_date - dt.timedelta(days=30)
            above_max_date = max_date + dt.timedelta(days=30)
            below_min_date, above_max_date = below_min_date.strftime("%Y-%m-%d"), above_max_date.strftime("%Y-%m-%d")
            
            # Random date within bounds
            random_date = dt.date(random.randint(min_date.year, max_date.year), random.randint(1, 12), random.randint(1, 28))
            random_date = random_date.strftime("%Y-%m-%d")
            
            
            # ************* Tests *************
        
            # Default
            try:
                df = retriever.retrieve(dataset=key, date=None, wavelength_bounds=None, max_queries=10)
                self.assertGreater(len(df), 0)
            except ValueError:
                self.fail()
            
            # Specific date within bounds
            try:    
                df = retriever.retrieve(dataset=key, date=random_date, wavelength_bounds=None, max_queries=10)
                self.assertGreater(len(df), 0) 
            except ValueError:
                self.fail()
            
            # Specific date below minimum (should raise ValueError)
            try:
                df = retriever.retrieve(dataset=key, date=below_min_date, wavelength_bounds=None, max_queries=10)
                self.assertTrue(True)
            except ValueError:
                self.fail()
            
            # Specific date above maximum (should raise ValueError)
            try:
                df = retriever.retrieve(dataset=key, date=above_max_date, wavelength_bounds=None, max_queries=10)
                self.assertTrue(True)
            except ValueError:
                self.fail()

            # Lower and upper wavelength bounds
            try:
                df = retriever.retrieve(dataset=key, date=None, wavelength_bounds=[100, 200], max_queries=10)
                self.assertGreater(len(df), 0)
            except ValueError:
                pass
            
            # Lower wavelength bound only
            try:
                df = retriever.retrieve(dataset=key, date=None, wavelength_bounds=[None, 200], max_queries=10)
                self.assertGreater(len(df), 0)
            except ValueError:
                pass
            
            # Upper wavelength bound only
            try:
                df = retriever.retrieve(dataset=key, date=None, wavelength_bounds=[100, None], max_queries=10)
                self.assertGreater(len(df), 0)
            except ValueError:
                pass
            
            # Lower wavelength bound greater than upper wavelength bound (should raise ValueError)
            try:
                df = retriever.retrieve(dataset=key, date=None, wavelength_bounds=[300, 200], max_queries=10)
                self.assertTrue(True)
            except ValueError:
                self.fail()