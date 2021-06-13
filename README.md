# Double Pendulum Fractal
A program that simulates many double pendulums on the GPU to produce a fractal image. Written in Python 3.6, and requires PyCuda, Tkinter, NumPy, and CUDA 8+. The double pendulum is modeled as two point masses connected by rigid, massless rods.

# Usage
To create a fractal image, run `src/python/gpu/double_pendulum_fractal_time_till_flip_interactive.py` or `src/python/gpu/double_pendulum_chaos_amount_fractal_interactive.py` from the top directory. Images are saved to an "interactive" directory. The PYTHONPATH environmental variable needs to include the `src\python` directory. 

Example commands to run the `double_pendulum_chaos_amount_fractal_interactive.py` program from a Cygwin command line, where the repository is cloned into C:\tmp:

    cd /cygdrive/c/tmp/double-pendulum-fractal
    export PYTHONPATH="C:\tmp\double-pendulum-fractal;C:\tmp\double-pendulum-fractal\src\python"
    python ./src/python/gpu/double_pendulum_chaos_amount_fractal_interactive.py
 
The `double_pendulum_fractal_time_till_flip_interactive.py` program colors each pixel based on how much time it took the first pendulum to flip. 

The `double_pendulum_chaos_amount_fractal_interactive.py` program colors each pixel based on how "chaotic" each initial configuration is. The "amount of chaos" for a pendulum is calculated by the average of the absoulte value of the difference between the final angle positions of the pendulum and its four neighbors to the top, bottom, left, and right.

* To zoom in or out, move the mouse over the location where you want to zoom and press the `z` key to zoom-in, or the `x` key to zoom-out. Zooming requires recomputing the entire image.

* Press `a` to increase the amount of time the pendulums are simulated, and `s` to decrease the time.

* Press `q` to decrease the time step or tolerance used in the simulations, and `w` to increase it.

* On Windows the TdrDelay registry value may need to be increased so that the CUDA kernel doesn't timeout when doing the computations.

* The amount of anti-aliasing can be adjusted by changing the value in the `set_anti_aliasing_amount` method.

* The `useDoublePrecision` variable can be set to `true` in order to use double-precision in the calculations, but doing so may also increase computation time significantly depending on what GPU is used. Enabling double-precision calculations allows for deeper zooms.

* The `src/python/gpu/double_pendulum_fractal_generate_images.py` program can be used to just generate images non-interactively, and generate zoom-in videos.

## Large images
### Time till flip
<p align="center">
  <img src="https://raw.githubusercontent.com/tryabin/double-pendulum-fractal/master/double%20pendulum%20fractal.png" alt="fractal image example" width="500" height="1000"/>
 </p>
 
 ### Amount of chaos (low energy)
 <p align="center">
  <img src="https://raw.githubusercontent.com/tryabin/double-pendulum-fractal/master/double%20pendulum%20fractal%20chaos%20amount%20low%20energy.png" alt="fractal image example" width="1000" height="1000"/>
</p>
