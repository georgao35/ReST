# setting CUDA devices
export CUDA_VISIBLE_DEVICES=0
# train half cheetah
python main.py --env HalfCheetah --expname halfcheetah --epochs 50 --num_skills 10
# train hopper
python main.py --env Hopper --expname hopper --epochs 50 --num_skills 10
# train walker
python main.py --env Walker2d --expname walker --epochs 50 --num_skills 10