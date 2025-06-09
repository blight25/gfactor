import unittest

from gfactor.querying.NISTQuerying import NISTRetriever
from gfactor.main.gfactoratomic import AtomicData

import numpy as np
import random

from pathlib import Path

class TestAtomic(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.required_cols = {"obs_wl(A)" : float, 
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
        
        self.optional_cols = {"unc_obs_wl": float}
        self.atomic_dir = Path("./data/atomic")
        self.atomic_dir.mkdir(parents=True, exist_ok=True)
        self.retriever = NISTRetriever()

        self.elements = [
                        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ar', 'Ca', 
                        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Co', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 
                        'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'I', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 
                        'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 
                        'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Pa', 'Th', 'Np', 'U', 'Am', 'Pu', 'Cm', 
                        'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Bh', 'Sg', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 
                        'Lv', 'Ts', 'Og'
                        ]

        # Known to fail: 'U' yields ValueError, all other queries proceed but produce invalid DataFrames
        self.problem_elements = {'Mt', 'At', 'Fm', 'U', 'Rf', 'Cf', 'Pa', 'No', 'Sg', 'Am', 'Pr', 'Nb', 
                                 'Lr', 'Rn', 'Lv', 'Zr', 'Os', 'Hs', 'Og', 'Mc', 'Cm', 'Fl', 'Cn', 'Db', 
                                 'Bk', 'Re', 'Rg', 'Po', 'Tb', 'Ds', 'Es', 'Th', 'Pm', 'Se', 'Bh', 'Nh', 
                                 'Ts', 'Md', 'Pu', 'Np'}


    def test_retrieve(self):

        # Loop through random subset of elements (full query is infeasible for reasonable runtimes)
        elements = random.sample(self.elements, k=20)

        for el in elements:
            if el in self.problem_elements:
                with self.assertRaises(ValueError):
                    self.retriever.retrieve(elements=[el], ionized=False, save_dir=None)
            else:
                for ionization in (True, False):
                    results = self.retriever.retrieve(elements=[el], ionized=ionization, save_dir=None)
                    df = results[el]
                    for col in self.required_cols:
                        self.assertIn(col, df.columns)
                        dtype = df[col].dtype
                        # Map NumPy dtype to Python type
                        python_type = float if np.issubdtype(dtype, np.floating) else int if np.issubdtype(dtype, np.int_) else str if np.issubdtype(dtype, np.str_) else None
                        self.assertIsNotNone(python_type)
                        self.assertEqual(python_type, self.required_cols[col])
                        
                    for col in self.optional_cols:
                        if col in df.columns:
                            dtype = df[col].dtype
                            # Map NumPy dtype to Python type
                            python_type = float if np.issubdtype(dtype, np.floating) else int if np.issubdtype(dtype, np.int_) else str if np.issubdtype(dtype, np.str_) else None
                            self.assertIsNotNone(python_type)
                            self.assertEqual(python_type, self.optional_cols[col])


    def test_load(self):

        # Path
        if not self.atomic_dir.exists():
            raise ValueError(f"Please run 'extract' in NISTQuerying.py, with save directory {self.atomic_dir}, "
                             + "before testing pre-saved loading behavior")

        # Loop through all elements
        for element in self.elements:
            if element in self.problem_elements:
                try:
                    AtomicData.load_nist(elements=[element])
                except ValueError:
                    continue
                self.fail(f"Value Error should have been raised for known problematic element {element}")
            
            else:
                for ionization in (True, False):
                    df = AtomicData.load_nist(elements=[element], ionized=ionization, data_dir=self.atomic_dir)
                    for col in self.required_cols:
                        if col not in df.columns:
                            self.fail(f"Required column {col} is missing for element {element}")
                        dtype = df[col].dtype
                        # Map NumPy dtype to Python type
                        python_type = float if np.issubdtype(dtype, np.floating) else int if np.issubdtype(dtype, np.int_) else str if np.issubdtype(dtype, np.str_) else None
                        if python_type is None:
                            self.fail(f"Unexpected dtype {dtype} for column '{col}'")
                        elif python_type != self.required_cols[col]:
                            self.fail(f"data type of {dtype} for '{col}' does not match expected data type of {self.required_cols[col]}")
                        
                    for col in self.optional_cols:
                        if col in df.columns:
                            dtype = df[col].dtype
                            # Map NumPy dtype to Python type
                            python_type = float if np.issubdtype(dtype, np.floating) else int if np.issubdtype(dtype, np.int_) else str if np.issubdtype(dtype, np.str_) else None
                            if python_type is None:
                                self.fail(f"Unexpected dtype {dtype} for column '{col}'")
                            elif python_type != self.optional_cols[col]:
                                self.fail(f"data type of {dtype} for '{col}' does not match expected data type of {self.required_cols[col]}")
            
        # Passed
        self.assertTrue(True)