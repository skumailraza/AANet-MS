#srun -K --ntasks=1 --gpus-per-task=2 --cpus-per-gpu=2 -p RTX6000 --mem=64000 \
#  --container-mounts=/netscratch:/netscratch,/ds-av:/ds-av,/netscratch/kraza/aanetms:/netscratch/kraza/aanetms \
#  --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh  \
#  --container-workdir=/netscratch/kraza/aanetms \
#  --container-mount-home \
#  --export="WANDB_API_KEY=406ca7642d853cdfbad965c078cf5c240a43a99e" \
#  sh scripts/aanet+_train.sh

  srun -K --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=4 -p V100-16GB  \
  --container-mounts=/netscratch:/netscratch,/ds-av:/ds-av,/netscratch/kraza/aanetms:/netscratch/kraza/aanetms \
  --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh  \
  --container-workdir=/netscratch/kraza/aanetms \
  --container-mount-home \
  sh scripts/aanet+_timing.sh