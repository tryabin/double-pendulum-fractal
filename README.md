# Double Pendulum Fractal
A program that simulates many double pendulums on the GPU to produce a fractal image. Written in Python 3.6, and requires PyCuda, Tkinter, NumPy, and CUDA 8+.

# Usage
* To create the fractal image, run `src/cuda/double_pendulum_fractal_interactive.py` from the top directory. Images are saved to the "interactive" directory. 

* To zoom in or out, move the mouse over the location where you want to zoom and press the `z` key to zoom-in, or the `x` key to zoom-out. Zooming requires recomputing the entire image.

* Press `a` to increase the amount of time the pendulums are simulated, and `s` to decrease the time.

* Press `q` to decrease the time step used in the simulations, and `w` to increase it.

* On Windows the TdrDelay registry value may need to be increased so that the CUDA kernel doesn't timeout when doing the computations.

* The amount of anti-aliasing can be adjusted by changing the value in the `set_anti_aliasing_amount` method.

* The `useDoublePrecision` variable can be set to `true` in order to use double-precision in the calculations, but doing so may also increase computation time significantly depending on what GPU is used. Enabling double-precision calculations allows for deeper zooms.

* The `src/cuda/double_pendulum_fractal_generate_images.py` program can be used to just generate images non-interactively.

<p align="center">
  <img src="https://raw.githubusercontent.com/tryabin/double-pendulum-fractal/master/double%20pendulum%20fractal.png" alt="fractal image example" width="500" height="1000"/>
  <img src="https://raw.githubusercontent.com/tryabin/double-pendulum-fractal/master/double%20pendulum%20fractal%20chaos%20amount%20low%20energy.png" alt="fractal image example" width="1000" height="1000"/>
</p>
