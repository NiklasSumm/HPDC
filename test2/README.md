# BitonicSortGPU

The implementation of a CUDA Bitonic sort of the Group 1 as a part of the ACA CUDA Homework.

## Instructions to run

1. Login to csg-headnode
2. Copy the code to the cluster and go to the directory
3. Activate the CUDA installation using

    ``` shell
    $ spack load cuda@12.6
    ```

4. Build source files using

    ``` shell
    $ bash build.sh
    ```

5. Run the code with

    ``` shell
    $ srun -p exercise-gpu --gres=gpu:1 python3 main.py
    ```

6. The application prints the results in the console and plots the benchmarked 
   results in the `comparison_plot.png` file.
   
## Example output

``` shell
(base) gpu01@csg-headnode:~/walli/BitonicSortGPU$ srun -p exercise-gpu --gres=gpu:1 python3 main.py
CUDA module loaded successfully!
=== Bitonic Sort Benchmark ===
       N   NumPy (ms)     CPU (ms)     GPU (ms)
--------  ----------  ----------  ----------
► Test N = 1024
    1024        0.006        0.028        0.340
► Test N = 2048
    2048        0.010        0.088        0.352
► Test N = 4096
    4096        0.019        0.210        0.379
► Test N = 8192
    8192        0.038        0.466        0.416
► Test N = 16384
   16384        0.079        0.981        0.467
► Test N = 32768
   32768        0.168        2.119        0.546
► Test N = 65536
   65536        0.379        4.503        0.649
► Test N = 131072
  131072        0.811        9.539        0.855
► Test N = 262144
  262144        1.674       20.201        1.202
► Test N = 524288
  524288        3.488       42.110        2.098
► Test N = 1048576
 1048576        7.286       88.379        2.930
► Test N = 2097152
 2097152       15.973      187.652        6.187
► Test N = 4194304
 4194304       32.935      390.703       11.686
► Test N = 8388608
 8388608       73.323      819.066       30.518
► Test N = 16777216
16777216      150.465     1710.524       56.868
► Test N = 33554432
33554432      309.873     3511.363      118.655

Plot saved as comparison_plot.png
```

