for alg in fedprox scaffold fednova
do
    python experiments.py --model=simple-cnn \
        --dataset=cifar10 \
        --alg=$alg \
        --lr=0.01 \
        --batch-size=64 \
        --epochs=10 \
        --n_parties=3 \
        --mu=0.01 \
        --rho=0.9 \
        --comm_round=100 \
        --partition=noniid-labeldir \
        --beta=0.5\
        --device='cuda:0'\
        --datadir='./data/' \
        --logdir='./logs/' \
        --noise=0 \
        --sample=1 \
        --init_seed=0
done