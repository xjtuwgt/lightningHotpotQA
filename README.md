python -m torch.distributed.launch --nproc_per_node=4 jdattrain.py --config_file configs/train.largerobertatest.json

python -m torch.distributed.launch --nproc_per_node=4 jdtrain.py --config_file configs/train.largeroberta.example.json

python lightningtrain.py --config_file configs/train.largeroberta.lightning.json