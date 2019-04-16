#!/bin/bash

# #cartpole experiments
# python main.py --env-name CartPoleVT-v0 --num-workers 8 --fast-lr 0.05 --max-kl 0.01 --fast-batch-size 10 --meta-batch-size 80 --num-layers 2 --hidden-size 100 --num-batches 100 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-cartpole-vt-nb100/mbs80fbs10 --device cuda

# python main.py --env-name CartPoleVT-v0 --num-workers 8 --fast-lr 0.05 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 100 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-cartpole-vt-nb100/mbs40fbs20 --device cuda

# python main.py --env-name CartPoleVT-v0 --num-workers 8 --fast-lr 0.05 --max-kl 0.01 --fast-batch-size 40 --meta-batch-size 20 --num-layers 2 --hidden-size 100 --num-batches 100 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-cartpole-vt-nb100/mbs20fbs40 --device cuda



# #acrobot experiments
# python main.py --env-name AcrobotVT-v1 --num-workers 8 --fast-lr 0.05 --max-kl 0.01 --fast-batch-size 10 --meta-batch-size 80 --num-layers 2 --hidden-size 100 --num-batches 100 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-acrobot-vt-nb100/mbs80fbs10 --device cuda

# python main.py --env-name AcrobotVT-v1 --num-workers 8 --fast-lr 0.05 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 100 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-acrobot-vt-nb100/mbs40fbs20 --device cuda

# python main.py --env-name AcrobotVT-v1 --num-workers 8 --fast-lr 0.05 --max-kl 0.01 --fast-batch-size 40 --meta-batch-size 20 --num-layers 2 --hidden-size 100 --num-batches 100 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-acrobot-vt-nb100/mbs20fbs40 --device cuda



# #continuous mountain car
# python main.py --env-name MountainCarContinuousVT-v0 --num-workers 8 --fast-lr 0.01 --max-kl 0.01 --fast-batch-size 10 --meta-batch-size 80 --num-layers 2 --hidden-size 100 --num-batches 20 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-cont-mount-car-vt-nb20/mbs80fbs10 --device cuda

# python main.py --env-name MountainCarContinuousVT-v0 --num-workers 8 --fast-lr 0.01 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 20 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-cont-mount-car-vt-nb20/mbs40fbs20 --device cuda

# python main.py --env-name MountainCarContinuousVT-v0 --num-workers 8 --fast-lr 0.01 --max-kl 0.01 --fast-batch-size 40 --meta-batch-size 20 --num-layers 2 --hidden-size 100 --num-batches 20 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-cont-mount-car-vt-nb20/mbs20fbs40 --device cuda






#kshot experiments

# #cartpole experiments
# python main.py --env-name CartPoleVT-v0 --num-workers 8 --fast-lr 0.025 --max-kl 0.01 --fast-batch-size 10 --meta-batch-size 80 --num-layers 2 --hidden-size 100 --num-batches 5 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-cartpole-vt-nb100/mbs80fbs10 --device cuda --exp-type KSHOT --K-shot-num-tasks 40

# python main.py --env-name CartPoleVT-v0 --num-workers 8 --fast-lr 0.025 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 5 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-cartpole-vt-nb100/mbs40fbs20 --device cuda --exp-type KSHOT --K-shot-num-tasks 40

# python main.py --env-name CartPoleVT-v0 --num-workers 8 --fast-lr 0.025 --max-kl 0.01 --fast-batch-size 40 --meta-batch-size 20 --num-layers 2 --hidden-size 100 --num-batches 5 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-cartpole-vt-nb100/mbs20fbs40 --device cuda --exp-type KSHOT --K-shot-num-tasks 40



# #acrobot experiments
# python main.py --env-name AcrobotVT-v1 --num-workers 8 --fast-lr 0.025 --max-kl 0.01 --fast-batch-size 10 --meta-batch-size 80 --num-layers 2 --hidden-size 100 --num-batches 10 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-acrobot-vt-nb100/mbs80fbs10 --device cuda --exp-type KSHOT --K-shot-num-tasks 40

# python main.py --env-name AcrobotVT-v1 --num-workers 8 --fast-lr 0.025 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 10 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-acrobot-vt-nb100/mbs40fbs20 --device cuda --exp-type KSHOT --K-shot-num-tasks 40

# python main.py --env-name AcrobotVT-v1 --num-workers 8 --fast-lr 0.025 --max-kl 0.01 --fast-batch-size 40 --meta-batch-size 20 --num-layers 2 --hidden-size 100 --num-batches 10 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-acrobot-vt-nb100/mbs20fbs40 --device cuda --exp-type KSHOT --K-shot-num-tasks 40



#continuous mountain car
python main.py --env-name MountainCarContinuousVT-v0 --num-workers 8 --fast-lr 0.01 --max-kl 0.01 --fast-batch-size 10 --meta-batch-size 80 --num-layers 2 --hidden-size 100 --num-batches 20 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-cont-mount-car-vt-nb20/mbs80fbs10 --device cuda --exp-type KSHOT --K-shot-num-tasks 40

python main.py --env-name MountainCarContinuousVT-v0 --num-workers 8 --fast-lr 0.01 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 20 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-cont-mount-car-vt-nb20/mbs40fbs20 --device cuda --exp-type KSHOT --K-shot-num-tasks 40

python main.py --env-name MountainCarContinuousVT-v0 --num-workers 8 --fast-lr 0.01 --max-kl 0.01 --fast-batch-size 40 --meta-batch-size 20 --num-layers 2 --hidden-size 100 --num-batches 20 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-cont-mount-car-vt-nb20/mbs20fbs40 --device cuda --exp-type KSHOT --K-shot-num-tasks 40
