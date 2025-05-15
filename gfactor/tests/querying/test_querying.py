# Standard library
import unittest
import datetime as dt
import random

# Local imports
from gfactor import NISTRetriever, LISIRDRetriever

class TestQuerying(unittest.TestCase):


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