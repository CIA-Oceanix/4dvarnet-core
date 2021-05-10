# Profiling

## Usage
A flag profile is set just before the training to toggle profiling, results are saved in `tb_profile`directory

## Profile dashboard

In order to access the profile dashboard:
- Launch profiling or download [result files](#results) the files to a directory `tb_profile`

- install the tensorboard extension:
  `pip install torch_tb_profiler`

- launch the dashboard
  `tensorboard --logdir tb_profile`

- Read the doc to see how to interpret
  dashboard : https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html#use-tensorboard-to-view-results-and-analyze-performance


## Results
I used the pytorch profiler on one epoch with 4 gpus and 20 grad iterations per forward step.

You can access the detailed files for 1 gpu here:
- training step profile
  files : https://s3.wasabisys.com/melody/4dvarnet-profile-res/r13i1n4_33112.1620647659266.pt.trace.json
- training step profile
  files : https://s3.wasabisys.com/melody/4dvarnet-profile-res/r13i1n4_33112.1620647869782.pt.trace.json
