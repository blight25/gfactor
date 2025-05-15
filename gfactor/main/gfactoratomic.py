# Standard libraries
from pathlib import Path
from astropy import units as u
from astropy.table import QTable, vstack

from typing import List, Dict, Union

import pandas as pd

from gfactor.querying.NISTQuerying import NISTRetriever


class AtomicData:
        
    elements_by_mass = {
        'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81, 'C': 12.011, 
        'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.18, 'Na': 22.99, 'Mg': 24.305, 
        'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'K': 39.098, 
        'Ar': 39.948, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 
        'Mn': 54.938, 'Fe': 55.845, 'Ni': 58.693, 'Co': 58.933, 'Cu': 63.546, 'Zn': 65.38, 
        'Ga': 69.723, 'Ge': 72.63, 'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798, 
        'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.95, 
        'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 
        'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'I': 126.9, 'Te': 127.6, 'Xe': 131.29, 
        'Cs': 132.91, 'Ba': 137.33, 'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 
        'Pm': 145.0, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.5, 
        'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97, 'Hf': 178.49, 
        'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08, 
        'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Po': 209.0, 
        'At': 210.0, 'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Pa': 231.04, 
        'Th': 232.04, 'Np': 237.0, 'U': 238.03, 'Am': 243.0, 'Pu': 244.0, 'Cm': 247.0, 
        'Bk': 247.0, 'Cf': 251.0, 'Es': 252.0, 'Fm': 257.0, 'Md': 258.0, 'No': 259.0, 
        'Lr': 262.0, 'Rf': 267.0, 'Db': 270.0, 'Bh': 270.0, 'Sg': 271.0, 'Hs': 277.0, 
        'Mt': 278.0, 'Ds': 281.0, 'Rg': 282.0, 'Cn': 285.0, 'Nh': 286.0, 'Fl': 289.0, 
        'Mc': 290.0, 'Lv': 293.0, 'Ts': 294.0, 'Og': 294.0
        }


    # Known to fail: 'U' yields ValueError, all other queries proceed but produce invalid DataFrames
    problem_elements = {'Mt', 'At', 'Fm', 'U', 'Rf', 'Cf', 'Pa', 'No', 'Sg', 'Am', 'Pr', 'Nb', 
                                'Lr', 'Rn', 'Lv', 'Zr', 'Os', 'Hs', 'Og', 'Mc', 'Cm', 'Fl', 'Cn', 'Db', 
                                'Bk', 'Re', 'Rg', 'Po', 'Tb', 'Ds', 'Es', 'Th', 'Pm', 'Se', 'Bh', 'Nh', 
                                'Ts', 'Md', 'Pu', 'Np'}
    
    def __init__(self):

        """AtomicData constructor"""
        
        self.retriever = NISTRetriever()


    def load_nist(elements: List[str], ionized: bool = False, data_dir=None) -> QTable:
        
        """
        Load NIST data for each element in `elements`.

        Parameters
        ----------
        elements : List[str]
            List of element symbols, e.g. ["H","C","O"].
        ionized : bool, optional
            indicates whether or not fetched data should incude ionized transitions
        data_dir: str, optional
            If provided, searches for pre-saved files in this directory before attempting to 
            query NIST.

        Returns
        -------
        combined: a single QTable of all elements stacked together.
        """

        tables: Dict[str, QTable] = {}
        retriever = NISTRetriever()

        for idx, el in enumerate(elements):

            # Confirm element data available
            if el not in AtomicData.elements_by_mass:
                raise ValueError(f"{el} is not a recognizeable element")
            elif el in AtomicData.problem_elements:
                raise ValueError(f"No eligible data for element {el}")
            
            # Pick the per-element file
            if data_dir:
                if ionized:
                    path = Path(f"{data_dir}/ionized/{el}_ionized.csv")
                else:
                    path = Path(f"{data_dir}/{el}.csv")
                exists = path.exists()
            else:
                exists = False

            # Read or retrieve
            if exists:
                df = pd.read_csv(path)
            else:
                df = retriever.retrieve(elements=[el], ionized=ionized)
                if df is None:
                    raise ValueError(f"No eligible data for element {el}")

            # Convert to QTable
            qt = QTable.from_pandas(df)
            for col in qt.colnames:
                
                # Identify units
                if "(" in col and col.endswith(")"):
                    _, unit_str = col.split("(", 1)
                    unit_str = unit_str[:-1]  # strip trailing ')'
                    if "^" in unit_str:
                        base, pow_str = unit_str.split("^", 1)
                        power = int(pow_str)
                    else:
                        base, power = unit_str, 1

                    # Map to astropy units
                    unit_map = {"A": u.AA, "s": u.s, "eV": u.eV}
                    base_unit = unit_map.get(base)
                    if base_unit is not None:
                        qt[col].unit = base_unit ** power

            tables[el] = qt

        # Stack and sort by wavelength
        combined = vstack(list(tables.values()), metadata_conflicts="silent")
        combined.sort("obs_wl(A)")
        return combined


if __name__ == "__main__":
     nist_data = AtomicData.load_nist(elements=['H','C','O','S'])