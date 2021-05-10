# Profile report

## Config

I used the pytorch profiler on one epoch with 4 gpus and 20 grad iterations per forward step

## Profile dashboard

In order to access the profile dashboard:

- Download the files to a directory `profile_report`
  You can access the detailed files for 1 gpu here:
- training step profile
  files : https://s3.wasabisys.com/melody/4dvarnet-profile-res/r13i1n4_33112.1620647659266.pt.trace.json
- training step profile
  files : https://s3.wasabisys.com/melody/4dvarnet-profile-res/r13i1n4_33112.1620647869782.pt.trace.json

- install the tensorboard extension:
  `pip install torch_tb_profiler`

- launch the dashboard
  `tensorboard --logdir profile_report`

- Read the doc to see how to interpret
  dashboard : https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html#use-tensorboard-to-view-results-and-analyze-performance
