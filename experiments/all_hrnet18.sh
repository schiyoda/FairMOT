cd src
python train.py mot --exp_id all_hrnet --gpus 0 --batch_size 8 --reid_dim 128 --arch 'hrnet_18'  --num_workers 0 --batch_size 4
cd ..