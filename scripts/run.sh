

RES=$(sbatch --parsable scripts/run.slurm old_data) && echo $RES >> scripts/old_data.jobs 
RES=$(sbatch --parsable scripts/run.slurm swot_sst) && echo $RES >> scripts/swot_sst.jobs
RES=$(sbatch --parsable scripts/run.slurm swot) && echo $RES >> scripts/swot.jobs
RES=$(sbatch --parsable scripts/run.slurm roll) && echo $RES >> scripts/roll.jobs 
RES=$(sbatch --parsable scripts/run.slurm nad) && echo ${RES##* } >> scripts/nad.jobs 
RES=$(sbatch --parsable scripts/run.slurm nad) && echo $RES >> scripts/nad.jobs
RES=$(sbatch --parsable scripts/run.slurm nad_roll) && echo $RES >> scripts/nad_roll.jobs
RES=$(sbatch --parsable scripts/run.slurm old_data_sst) && echo $RES >> scripts/old_data_sst.jobs
RES=$(sbatch --parsable scripts/run.slurm nad_swot) && echo $RES >> scripts/nad_swot.jobs
RES=$(sbatch --parsable scripts/run.slurm roll_sst) && echo $RES >> scripts/roll_sst.jobs
RES=$(sbatch --parsable scripts/run.slurm nad_sst) && echo $RES >> scripts/nad_sst.jobs


sbatch scripts/test.slurm $(tail -n1 scripts/nad_swot.jobs) nad_swot >> scripts/nad_swot.test_jobs
sbatch scripts/test.slurm $(tail -n1 scripts/swot.jobs) swot >> scripts/swot.test_jobs
sbatch scripts/test.slurm $(tail -n1 scripts/old_data_sst.jobs) old_data_sst >> scripts/old_data_sst.test_jobs
sbatch scripts/test.slurm $(tail -n1 scripts/roll.jobs) roll >> scripts/roll.test_jobs
sbatch scripts/test.slurm $(tail -n1 scripts/swot_sst.jobs) swot_sst >> scripts/swot_sst.test_jobs
sbatch scripts/test.slurm $(tail -n1 scripts/nad_roll.jobs) nad_roll >> scripts/nad_roll.test_jobs
sbatch scripts/test.slurm $(tail -n1 scripts/nad_sst.jobs) nad_sst >> scripts/nad_sst.test_jobs
sbatch scripts/test.slurm $(tail -n1 scripts/nad.jobs) nad >> scripts/nad.test_jobs
sbatch scripts/test.slurm $(tail -n1 scripts/old_data.jobs) old_data >> scripts/old_data.test_jobs
sbatch scripts/test.slurm $(tail -n1 scripts/roll_sst.jobs) roll_sst >> scripts/roll_sst.test_jobs

tail -f *$(tail -n1 scripts/nad_swot.jobs)*
tail -f *$(tail -n1 scripts/swot.jobs)*
tail -f *$(tail -n1 scripts/old_data_sst.jobs)*
tail -f *$(tail -n1 scripts/roll.jobs)*
tail -f *$(tail -n1 scripts/swot_sst.jobs)*
tail -f *$(tail -n1 scripts/nad_roll.jobs)*
tail -f *$(tail -n1 scripts/nad_sst.jobs)*
tail -f *$(tail -n1 scripts/nad.jobs)*
tail -f *$(tail -n1 scripts/old_data.jobs)*
tail -f *$(tail -n1 scripts/roll_sst.jobs)*

