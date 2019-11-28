# Double Pendulum Fractal
A program that simulates many double pendulums on the GPU to produce a fractal image. Written in Python 3.6, and requires PyCuda, Tkinter, and NumPy.

To create the fractal image, run `src/cuda/double_pendulum_fractal.py` from the top directory. On Windows the TdrDelay registry value may need to be increased so that the CUDA kernel doesn't timeout when doing the computations.

The `useDoublePrecision` variable can be set to `true` in order to use double-precision in the calculations, but doing so may also increase computation time significantly depending on what GPU is used.

<p align="center">
  <img src="https://raw.githubusercontent.com/tryabin/double-pendulum-fractal/master/double%20pendulum%20fractal.png" alt="fractal image example"/>
</p>
