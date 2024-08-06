# PyRootMancer Documentation

Welcome to the official documentation for PyRootMancer, your ultimate companion for root phenotyping!

## Installation

To install PyRootMancer, you can use pip:

```
pip install pyrootmancer
```

PyRootMancer requires Python 3.6 or above.

## Getting Started

### Quick Example

Here's a quick example to get you started with PyRootMancer:

```python
import pyrootmancer
```
# Load your root data
root_data = pyrootmancer.load_data('path/to/your/root_data.csv')

# Analyze root morphology
morphology_results = pyrootmancer.analyze_morphology(root_data)

# Visualize the results
pyrootmancer.plot_morphology(morphology_results)

Detailed Usage

PyRootMancer provides several modules and functions to facilitate root phenotyping. Here are some key features:

    Data Loading: Load root data from various formats.
    Morphology Analysis: Analyze root morphology and growth patterns.
    Visualization: Visualize root data using matplotlib and interactive tools.
    Community and Collaboration: Join our community to contribute and improve PyRootMancer.

API Reference
pyrootmancer.load_data(file_path)

Load root data from a file.

    Parameters:
        file_path (str): Path to the root data file.

    Returns:
        root_data (DataFrame or array-like): Loaded root data.

pyrootmancer.analyze_morphology(root_data)

Analyze root morphology and growth patterns.

    Parameters:
        root_data (DataFrame or array-like): Root data to analyze.

    Returns:
        morphology_results (dict): Dictionary containing morphology analysis results.

pyrootmancer.plot_morphology(morphology_results)

Visualize morphology analysis results.

    Parameters:
        morphology_results (dict): Morphology analysis results to visualize.

    Returns:
        Visualization plot.