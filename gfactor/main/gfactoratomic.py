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
    'Mn': 54.938, 'Fe': 55.845
    }
     
    def __init__(self):

        """AtomicData constructor"""
        
        self.retriever = NISTRetriever()


    def load_nist(self, elements: List[str], ionized: bool = False, data_dir=None) -> QTable:
        
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

        # 1) Sort by atomic mass
        elements_sorted = sorted(elements, key=lambda el: AtomicData.elements_by_mass[el])

        tables: Dict[str, QTable] = {}
        retriever = NISTRetriever()

        for el in elements_sorted:
            # 2) Pick the per-element file
            if data_dir:
                if ionized:
                    path = Path(f"{data_dir}/ionized/{el}_ionized.csv")
                else:
                    path = Path(f"{data_dir}/{el}.csv")

            # 3) Read or retrieve
            if data_dir:
                if path.exists():
                    df = pd.read_csv(path)
            else:
                # retrieve expects a list of elements
                df = retriever.retrieve(elements=[el], ionized=ionized)
                if df is None:
                    raise ValueError(f"No eligible data for element {el}")

            # 4) Convert to QTable and fix units
            qt = QTable.from_pandas(df)
            for col in qt.colnames:
                if "(" in col and col.endswith(")"):
                    _, unit_str = col.split("(", 1)
                    unit_str = unit_str[:-1]  # strip trailing ')'
                    if "^" in unit_str:
                        base, pow_str = unit_str.split("^", 1)
                        power = int(pow_str)
                    else:
                        base, power = unit_str, 1

                    # map to astropy units
                    unit_map = {"A": u.AA, "s": u.s, "eV": u.eV}
                    base_unit = unit_map.get(base)
                    if base_unit is not None:
                        qt[col].unit = base_unit ** power

            tables[el] = qt

        # 5) stack and sort by wavelength
        combined = vstack(list(tables.values()), metadata_conflicts="silent")
        combined.sort("obs_wl(A)")
        return combined


if __name__ == "__main__":
     atomic_data = AtomicData()
     nist_data = atomic_data.load_nist(elements=['H','C','O','S'], join=True)