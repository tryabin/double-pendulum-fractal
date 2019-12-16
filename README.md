# Double Pendulum Fractal
A program that simulates many double pendulums on the GPU to produce a fractal image. Written in Python 3.6, and requires PyCuda, Tkinter, NumPy, and CUDA 8+.

# Usage
* To create the fractal image, run `src/cuda/double_pendulum_fractal.py` from the top directory. 

* To zoom in or out, move the mouse over the location where you want to zoom and press the `z` key to zoom-in, or the `x` key to zoom-out. Zooming requires recomputing the entire image.

* On Windows the TdrDelay registry value may need to be increased so that the CUDA kernel doesn't timeout when doing the computations.

* The `antiAliasingGridWidth` variable can be increased to improve anti-aliasing.

* The `useDoublePrecision` variable can be set to `true` in order to use double-precision in the calculations, but doing so may also increase computation time significantly depending on what GPU is used. Enabling double-precision calculations allows for deeper zooms.

<p align="center">
  <img src="https://raw.githubusercontent.com/tryabin/double-pendulum-fractal/master/double%20pendulum%20fractal.png" alt="fractal image example" width="500" height="1000"/>
</p>
