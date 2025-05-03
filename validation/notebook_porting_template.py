import nbformat
from typing import List

def create_nb_subset(file_loc, subset:List[int], save_loc):

    # Load the existing notebook
    with open(file_loc, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Extract cells
    subset_cells = nb.cells[subset[0]:2*subset[1]] # Weird bug where final index has to be multiplied by 2

    # Create a new notebook with the extracted cells
    new_nb = nbformat.v4.new_notebook()
    new_nb.cells = subset_cells

    # Save the new notebook
    with open(save_loc, 'w', encoding='utf-8') as f:
        nbformat.write(new_nb, f)

    print(f"New notebook '{save_loc}' created from cells {subset[0]}-{subset[1]}.")


if __name__ == "__main__":
    file_loc = './gfactor/ML/regression/fitting.ipynb'
    subset = [0, 18]
    save_loc = './gfactor/ML/regression/min_vs_max_testing.ipynb'
    create_nb_subset(file_loc, subset, save_loc)
    
    
    