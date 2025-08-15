# schlieren-imaging-project

## Overview

This project explores complex analysis and potential flow theory for airfoil and barrier shapes, with a focus on visualizing conformal mappings and comparing theoretical airfoil shapes to real-world data.

The code in the `complex_analysis` folder provides:
- Classes for representing and manipulating complex numbers and shapes (circles, airfoils, prisms) in the complex plane.
- Tools for applying the Joukowski transform to generate airfoil shapes from circles.
- Visualization utilities for comparing theoretical airfoils to 3D-printed (NACA) airfoil data.
- Plotting of potential flow fields around different barriers.

## Features

- **Airfoil Comparison Plotter:** Compare the Joukowski-transformed airfoil with a real, printed airfoil shape.
- **Potential Flow Plotter:** Visualize potential flow fields around various barriers (plane, prism, cylinder, airfoil).
- **Support for custom barriers and conformal maps.**
- **Example data:** Includes a CSV file with measured airfoil coordinates.

## Example Images

### Airfoil vs. Cylinder Example

![Airfoil vs Cylinder](complex_analysis/airfoil_cylinder_example.png)

### Plain Prism Example

![Plain Prism](complex_analysis/plain_prism_example.png)

## Usage

### Airfoil Comparison

To run the airfoil comparison plotter:

```python
from complex_analysis.examples import example_usage_airfoil_comparison
example_usage_airfoil_comparison()
```

### Potential Flow Plotter

To visualize potential flow around different barriers:

```python
from complex_analysis.examples import example_usage_potential_flow_plotter_1, example_usage_potential_flow_plotter_2
example_usage_potential_flow_plotter_1()  # Plane vs Prism
example_usage_potential_flow_plotter_2()  # Airfoil vs Cylinder
```

## Data

- [complex_analysis/printed.csv](complex_analysis/printed.csv): Contains measured coordinates of a real airfoil for comparison.

## Project Structure

- `complex_analysis/`: Complex analysis and visualization tools.
    - `airfoil_comparison_plotter.py`: Airfoil comparison plotting.
    - `barrier.py`: Barrier and conformal map classes.
    - `complex_numbers.py`: Complex number utilities.
    - `complex_plane.py`: Shape representations and transforms.
    - `potential_flow_plotter.py`: Potential flow visualization.
    - `examples.py`: Example usage scripts.
    - `airfoil_cylinder_example.png`, `plain_prism_example.png`: Example output images.
    - `printed.csv`: Airfoil measurement data.

---

*For more details, see the docstrings in