srun -K --ntasks=1 --gpus-per-task=8 --cpus-per-gpu=3 -p RTX2080Ti --mem=64000 \
  --container-mounts=/netscratch:/netscratch,/ds-av:/ds-av,/netscratch/kraza/aanetms:/netscratch/kraza/aanetms \
  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.01-py3.sqsh  \
  --container-workdir=/netscratch/kraza/aanetms \
  --container-mount-home \
  --export="WANDB_API_KEY=406ca7642d853cdfbad965c078cf5c240a43a99e" \
  sh scripts/aanet+_train.sh