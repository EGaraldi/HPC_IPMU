> [!TIP]
> If you have comments or suggestions to improve this page, please send them to `egaraldi_at_ipmu.jp` or - better - make a pull request directly to this repo

# High Performance Computing at Kavli IPMU

This page contains a set of useful tips and tricks for high-performance computing on the IPMU cluster. 

There are 5 available machines at IPMU:

- `idark`, the main computer cluster.
- `gw`, the previous computing cluster.
- `gfarm`, another older computing cluster
- `gpgpu`, a box with 8 GPUs.
- `igpu`, a newer machine with 20 GPUs.

These machines are managed by IPMU's IT team, who can be reached at  `it_at_ipmu.jp`. 

Technical details and specifications can be found in the [internal webpage](https://www.ipmu.jp/en/employees-internal/computing) (ask IT if you don't have access yet).


## Contents
 - [access](#accessing-the-machines) 
 - [check usage](#check-the-machine-usage)
 - [pyhton environment](#setting-up-a-python-environment)
 - [running jobs](#running-jobs-on-the-cluster)
 - [accessing nodes](#accessing-the-compute-nodes)
 - [using jupyter](#using-jupyter-and-port-forwarding)
 - [IDE](#connecting-your-ide)
 - [storage on idark](#where-to-work-on-idark) 
 - [globus endpoint](#globus-endpoint-on-idark) 
 - [igpu server](#igpu-server)

## Accessing the machines

The machines can only be accessed from within the campus intranet. To use them from home, you need to use [the IPMU VPN](https://www.ipmu.jp/en/employees-internal/computing#VPN).

You will need an account on the machines you wish to access. To get it, follow the instructions [on this page (section 3)](https://www.ipmu.jp/en/employees-internal/computing/cluster).

Once you're all set up you can connect to the servers with
```bash
$ ssh [username]@idark.ipmu.jp  # for idark
$ ssh [username]@192.168.156.68 # for gw
$ ssh [username]@gfarm.ipmu.jp  # for gfarm
$ ssh [username]@192.168.156.71 # for gpgpu
$ ssh [username]@192.168.156.50 # for igpu
```

## Check the machine usage

Once you ssh onto the cluster, you'll want to see what everyone else is doing.

The job manager on some machine (including idark) is [PBS](https://albertsk.org/wp-content/uploads/2011/12/pbs.pdf). To see what jobs are running, run

```bash
[username@idark ~]$ qstat
```

On other machines, the job manager is [slurm](https://slurm.schedmd.com/documentation.html). The equivalent command is

```bash
[username@gpgpu ~]$ squeue
```

It's important to know that there is no central system to allocate the GPUs on gpgpu, so you need to check which are available using the command

```bash
[username@gpgpu ~]$ nvidia-smi
```

## Setting up a Python environment

> [!TIP]
> The python version installed on the cluster is quite old by now. It is reccommended to install a more recent version using e.g. conda (see below).

Before running your own jobs, you may wish to set up your Python environment. Python 3 is already installed on the cluster, and there are two main approaches to managing your python installation: `conda` and `pyenv`. Choose whichever you prefer.

### Conda

> [!CAUTION]
> Recently (approx. 2024) Anaconda changed their pricing policy and can/will now charge users of medium-size organization if they use the default channel. Please make sure to do one of the following:
> - use [miniforge](https://github.com/conda-forge/miniforge), a community-maintained, truly free version of conda;
> - in conda, replace the `default` package channel with `conda-forge` by running: `conda config --prepend channels new_channel && conda config --remove channels default`

Conda is a python package and environment manager. Suppose you start a new project and want to use all of the up-to-date versions of your favorite python modules -- but want to keep older versions available for compatibility with previous projects. This is the rationale of conda. With conda you can create a library of separate python environments that you can activate anywhere on the cluster -- from the login node or compute nodes, either in interactive mode or directly inside your job scripts. 

The first time you log on to the cluster, run
```bash
[username@idark ~]$ conda init
```
This will add an initialization script to your `~/.bashrc` that is executed whenever you login to the cluster or allocate to a compute node. The next time you login, you will see `(base)` next to your credentials on the command line, indicating that your `base` conda environment is active. When you run python,
```bash
(base) [username@idark ~]$ python
```
and you will see that the version that has been activated is Python 3.8.10.

If you prefer, you can always turn off automatic activation of the environment by doing:
```bash
[username@idark ~]$ conda config --set auto_activate_base false
```
and instead manually activate the environment using 
```bash
[username@idark ~]$ source /home/anaconda3/bin/activate
```
or
```bash
[username@idark ~]$ conda activate base
```

To deactivate a conda environment, just run `conda deactivate`.

You can also create your own environments with conda with any specified python version:

```bash
[username@idark ~]$ conda create -n project-env python=3.9
```

This command creates a new python environment called `project-env` which runs the latest stable version of Python 3.9. Once you have created the new environment, you can view it and all other existing environments (including `base`) with `conda env list`. You can activate any of these environments as follows:
```bash
[username@idark ~]$ conda activate project-env
``` 
replacing `project-env` with the name of your environment. 

The default installation will be fairly bare-bones, but you can now start installing packages. To install numpy, for example, you can do:
```bash
(project-env) [username@idark ~]$ conda install numpy
```

If a package you want is unavailable from the conda package manager, you can use `pip` within a conda environment instead. Type `which pip` and you will see that it is a specific pip installation for your python environment. Note, however, that this can sometimes lead to conflicts; see [here](https://www.anaconda.com/blog/using-pip-in-a-conda-environment) for details. 

`numpy` is one of many useful Python packages. Wouldn't it be nice if there is a stack of all the useful scientific packages so that you wouldn't have to install them all separately and think about dependencies? Oh yeah:
```bash
(project-env) [username@idark ~]$ pip install scipy-stack
```

Later, we will see how to use this python environment for jobs. For more documentation on conda, including more management options and deletion of environments, check out the [conda docs](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### pyenv

Unlike conda, which is a full package manager, pyenv is a simpler tool that just manages your different python versions. The idea is to use pip as the package manager and python's built in virtual environment feature to separate dependencies for different projects.

pyenv does not come pre-installed on the cluster machines. To install it, we use the [pyenv-installer](https://github.com/pyenv/pyenv-installer.
). First run 

```bash
[username@idark ~]$ curl https://pyenv.run | bash
```

and then add the following lines to your bash profile `~/.bashrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

After restarting your terminal, pyenv should now be in your path and you can see the list of installable python versions:
```bash
[username@idark ~]$ pyenv install --list # see a list of installable versions
```
and then install one using
```bash
[username@idark ~]$ 
CPPFLAGS=-I/usr/include/openssl11 \
LDFLAGS=-L/usr/lib64/openssl11 \
PYTHON_CONFIGURE_OPTS="--enable-shared" \
pyenv install -v 3.11.3 # or something different
```

The new version will now show up if you run `pyenv versions`, and you can set it as your global python version using 

```bash
[username@idark ~]$ pyenv global 3.11.3

[username@idark ~]$ python --version
# Python 3.11.3

[username@idark ~]$ which python
# ~/.pyenv/shims/python

[username@idark ~]$ which pip
# ~/.pyenv/shims/pip
```

You can also set local python versions for different projects using `pyenv local`. Then to manage dependencies for different projects, you use python's built in `venv` feature. Here is a [tutorial](https://docs.python.org/3/tutorial/venv.html), and the tldr is as follows. Starting from your project folder, initialize a new virtual environment and activate it with

```bash
[username@idark project]$ python -m venv project-venv
[username@idark project]$ source project-venv/bin/activate
```

At which point you will see (project-venv) in the command prompt. If you now install dependencies or projects with pip, they will be installed in the local virtual environment.

## Running jobs on the cluster

Anytime you want to run code on the cluster, you should do so in a job. There are a lot of different types of jobs.

The basic pattern for a PBS job on idark is

```bash
#!/bin/bash
#PBS -N demo
#PBS -o /home/username/PBS/
#PBS -e /home/username/PBS/
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=0:0:30
#PBS -u username
#PBS -M username@ipmu.jp
#PBS -m ae
#PBS -q tiny

# activate python environment
source ~/.bashrc

#your commands here, e.g.
mpirun -np 2 my_parallel_code
```

You can then submit the job script with

```bash
qsub job.sh
```
and check its status with 

```bash
qstat -u <your username>
```
(more details on `qstat` can be found [here](https://www.jlab.org/hpc/PBS/qstat.html))

For a  job on gpgpu, make sure to first run `nvidia-smi` and identify GPUs which are not being used. Then a typical slurm file looks like

```
#!/bin/bash
#SBATCH --job-name=demo
#SBATCH --account=username
#SBATCH --output=/home/username/log/%j.out  
#SBATCH --error=/home/username/log/%j.err  
#SBATCH --time==0+00:01:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=6
#SBATCH --mail-user=username@ipmu.jp
#SBATCH --mail-type=END,FAIL

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=X # where X is the GPU id of an available GPU

# activate python environment
conda activate project-env
# for pyenv/virtualenv instead use
# source /home/username/project/project-venv/bin/activate

python my_program.py
```
> [!NOTE]
> Please use only one GPU at a time, to prevent congestion!

You can then submit the job script with

```bash
sbatch job.sh
```
and check its status with 

```bash
squeue -u <your username>
```
(more details on `squeue` can be found [here](https://slurm.schedmd.com/squeue.html))

### Fine-grained controls

#### MPI Tasks assignment

There are multiple ways to assign MPI tasks to physical CPUs. Two common ways are contiguous and round-robin assignment, i.e.

*round-robin assignment*
|  node  |  MPI tasks | 
---------|------------| 
| 1      |  0, 3, 6, 9, ...|  
| 2      |  1, 4, 7, 10, ...|  
| 3      |  2, 5, 8, 11, ...|  


*contiguous assignment*
|  node  |  MPI tasks | 
---------|------------| 
| 1      |  0, 1, 2, 3, ...|  
| 2      |  52, 53, 54, 55, ...| 
| 3      |  104, 105, 106, 107, ...| 

By default, PBS assigns MPI Tasks in a round-robin fashion. To force a contiguous assignment (as done by e.g. SLURM), use:

``` bash
#PBS -l select=3:ncpus=52:mem=4gb:mpiprocs=52
```
Notice the last part (`:mpiprocs=52`), which is redundant, but forces a contiguous assignment. 

## Accessing the compute nodes

On idark, you can directly access the compute node that is running your job. First identify the node by running

``` bash
[username@idark] $ qstat -nrt -u username
```
The node will have a name of the format `ansysNN` where `NN` in a 2-digit number between 01 and 40. 

Then use `rsh` to connect to it

```bash
[username@idark ~]$ rsh ansysNN
[username@ansysNN ~]$
```

## Using Jupyter and port forwarding

Using jupyter is a great way to make working remotely a little easier and more seamless.

You should always run jupyter in a job. Make a job script which runs the command 

```bash
jupyter lab --no-browser --ip=0.0.0.0 --port=X
```

where X is some port you choose, e.g 1337. Each port can only be used by one application at a time so make sure your jupyter session chooses a unique port number.

Now you just need to forward some port Y on your local computer to the remote port X. On gpgpu, this is as simple as running

```bash
$ ssh -NfL Y:localhost:X username@192.168.156.71 #for gpgpu
```
Now opening your browser to localhost:Y will give you direct access to jupyter on gpgpu. This same kind of port forwarding is used if you want to use MLFlow or tensorboard to track your ML experiments. 

On idark, you first need to find node your jupyter is running on by running `qstat -nrt -u username`.

Then on your local computer just run
```bash
$ ssh -NfL Y:ansysNN:X username@idark.ipmu.jp #for idark
```
where ansysNN is the node name. You'll now be able to access jupyter by pointing your browser to localhost:Y . The first time you connect to a jupyter session, it may prompt you for a connection token. You can get the token by connecting to the compute node using `rsh ansysNN` from the head node, and then running `jupyter server list`. You can use that token on the connection page to set a jupyter password which you can use to connect to future jupyter sessions on any node.


<!-- On gpgpu, to give GPUs to your jupyter instance you should put the jupyter lab command in a job script. -->

<!-- If you get an ssl error running jupyter lab, then either make sure you activate conda in your job script or if using pyenv find the missing libraries mentioned in the error message using `locate libssl.so.1.1`, copy library from the login node to a folder `/home/username/mylibs/`, and and then in your job script add the line `export LD_LIBRARY_PATH=/home/username/mylibs/:$LD_LIBRARY_PATH` .  -->

To register a python environment with jupyter, just activate the environment and then run
```bash
python -m ipykernel install --name project-venv --user
```
and you'll see a kernel with name project-venv next time you launch jupyter.

## Connecting your IDE 

Most IDEs you run on your local computer can be connected to the cluster to allow you to edit your files there seamlessly. For vscode, for example, follow the instructions [here](https://code.visualstudio.com/docs/remote/ssh).

> [!NOTE]
> New versions (>1.98) of vscode will refuse connection because the system libraries on idark are too old. To fix this (thanks, Yasuda-san!) add the following to your `.bashrc`:
> 
> ```bash
> export VSCODE_SERVER_CUSTOM_GLIBC_LINKER=/lustre/work/yasuda/ct_ng/toolchain-dir/x86_64-linux-gnu/x86_64-linux-gnu/sysroot/lib/ld-2.28.so
> export VSCODE_SERVER_CUSTOM_GLIBC_PATH=/lustre/work/yasuda/ct_ng/toolchain-dir/x86_64-linux-gnu/x86_64-linux-gnu/sysroot/lib
> export VSCODE_SERVER_PATCHELF_PATH=/lustre/work/yasuda/local/bin/patchelf
> ```


## Where to work on idark

The `/home` file system has very limited space and may rapidly get congested if too many users are working and storing files there. As with previous clusters, iDark has a designated file system for work:

    /lustre/work/username
    
The snag is that the output files from job scripts will not save to this file system -- could be quite inconvenient. This issue has been raised with IT, but for now you should create a directory in the `/home` file system (e.g. `/home/username/tmp/pbs/` where those files can be directed in the `#PBS -o` and `#PBS -e` lines of your job scripts.

## Globus endpoint on idark

[Globus](https://app.globus.org/) is a powerful tool to transfer large amounts of data in an automated, unsupervised, safe way. An exhaustive tutorial is available [on the lobus.org website](https://docs.globus.org/getting-started/users/). 

On `idark`, we have a public Globus endpoint installed, named `KIPMU-Collection`.

## IGPU server

The IGPU cluster is for GPU-focused work loads. It has 8+1 nodes with 4 V100 GPUs each. Contact `leander[dot]thiele[at]ipmu[dot]jp`.

In contrast to IDark, IGPU uses the Slurm scheduling system. 

Some tips:
- Use `ssh igpuNN` to login to the compute nodes. There you can use `nvidia-smi` or `gpustat` (recommended) to monitor GPU utilization.
- In machine learning applications, it is sometimes nontrivial to use the GPUs efficiently. Consider using multiprocessing on CPUs to improve GPU utilization.
- It has been observed that a bug in Slurm puts jobs on the head node. Use `#SBATCH --nodelist=igpuNN` or similar to prevent such behavior.
- I recommend the `micromamba` environment manager (faster and free compared to `conda`).

Using the Slurm system:
- `sinfo` to get an overview of node status
- `squeue` to see queued and running jobs

The following is a minimal job submission script illustrating the main things.
It contains some peculiarities for making PyTorch distributed training work across multiple nodes.
```bash
#!/usr/bin/env bash

#SBATCH --job-name=fnldmA
#SBATCH -N 4
#SBATCH --ntasks=16
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH -t 96:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# from tutorial at https://pytorch-geometric.readthedocs.io/en/latest/tutorial/multi_node_multi_gpu_vanilla.html
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

# neccessary for the GLOO cpu-to-cpu communication protocol to work
export GLOO_SOCKET_IFNAME=eno3

# this is specific to the micromamba manager
source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate void_finding

# use srun to launch ntasks jobs
srun python train.py --config=./configs/fnldmA.yaml
```
