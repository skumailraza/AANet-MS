#conda env create -f environment.yml

#conda init bash
#source /root/.bashrc
#conda activate aanet
pip install --upgrade wandb
pip install --upgrade tensorboard
pip install -U scikit-image

cd nets/deform_conv
sh build.sh
