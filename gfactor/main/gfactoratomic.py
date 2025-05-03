# Standard libraries
import string
from typing import List
from pathlib import Path
from astropy import units as u
from astropy.units import Quantity
from astropy.table import QTable

import pandas as pd

from itertools import combinations

from gfactor.querying.NISTQuerying import NISTRetriever


class AtomicData:

    subgroups = {'HCOS': ['H', 'C', 'O', 'S']}
        
    elements_by_mass = {
    'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81, 'C': 12.011, 
    'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.18, 'Na': 22.99, 'Mg': 24.305, 
    'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'K': 39.098, 
    'Ar': 39.948, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 
    'Mn': 54.938, 'Fe': 55.845
    }
     
    def __init__(self):
        
        self.retriever = NISTRetriever()


    def load_nist(elements: List[str], wavelength_bounds: List[int], ionized=False):

        elements.sort(key = lambda x: AtomicData.elements_by_mass[x]) # Sort list
        elements_str = ''.join(elements)

        def find_subsets(input_list):
                subsets = []
                for r in range(1, len(input_list) + 1):
                    subsets.extend(combinations(input_list, r))
                return subsets
        
        subsets = {*find_subsets(AtomicData.subgroups['HCOS'])}

        file = None
        if tuple(elements) in subsets:
            if ionized:     
                file = f"./atomic_data/ionized/{elements_str}_{wavelength_bounds[0]}-{wavelength_bounds[1]}_ionized.csv"
            else:
                file = f"./atomic_data/standard/{elements_str}_{wavelength_bounds[0]}-{wavelength_bounds[1]}.csv"
            file = Path(file)
        
        if file and file.exists():
            nist_data = pd.read_csv(file)
            
        else:
            nist_retriever = NISTRetriever()
            nist_data = nist_retriever.retrieve(wavelength_bounds=wavelength_bounds, elements=elements, ionized=ionized)
        
        nist_data = QTable.from_pandas(nist_data)
        
        for col in nist_data.colnames:
            parts = col.split('(')
            power = 1
            if len(parts) > 1:
                var, unit = parts
                unit = unit[:-1]
                parts = unit.split('^')
                if len(parts) > 1:
                    unit, power = parts
                match unit:
                    case 'A':
                        unit = u.AA
                    case 's':
                        unit = u.s
                    case 'eV':
                        unit = u.eV
                nist_data[col].unit = unit**int(power)
                
        return nist_data


if __name__ == "__main__":
     atomic_data = AtomicData()
     nist_data = atomic_data.load_nist(wavelength_bounds=[800, 7000], elements=['H','C', 'O', 'S'])