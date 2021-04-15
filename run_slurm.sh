srun -K --ntasks=1 --gpus-per-task=3 --cpus-per-gpu=3 -p V100-32GB --mem=64000 \
  --container-mounts=/netscratch:/netscratch,/ds-av:/ds-av,/netscratch/kraza/aanetms:/netscratch/kraza/aanetms \
  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_19.06-py3.sqsh  \
  --container-workdir=/netscratch/kraza/aanetms \
  --container-mount-home \
  --export="WANDB_API_KEY=406ca7642d853cdfbad965c078cf5c240a43a99e" \
  sh scripts/aanet+_train.sh