# HONEST transfer + combined eval: cyber_3 ↔ refusal_gemma

Train on `split=='train'` only; eval on `split=='test'`.
LR (C=1.0 single-layer, C=0.1 13-layer concat).


## Single-layer mean-pool transfer

| Layer | cyber_3-only → cyber_3 | cyber_3-only → refusal | refusal-only → refusal | refusal-only → cyber_3 | combined → cyber_3 | combined → refusal |
|---|---:|---:|---:|---:|---:|---:|
| L30 | 0.9652 | 0.8068 | 0.9125 | 0.7868 | 0.9520 | 0.8640 |
| L35 | 0.9727 | 0.8048 | 0.9126 | 0.7946 | 0.9557 | 0.8616 |
| L40 | 0.9695 | 0.8113 | 0.9269 | 0.7687 | 0.9513 | 0.8704 |
| L45 | 0.9681 | 0.7716 | 0.8916 | 0.7574 | 0.9494 | 0.8401 |

## Single-layer last-token transfer

| Layer | cyber_3-only → cyber_3 | cyber_3-only → refusal | refusal-only → refusal | refusal-only → cyber_3 | combined → cyber_3 | combined → refusal |
|---|---:|---:|---:|---:|---:|---:|
| L30 | 0.9147 | 0.7877 | 0.8991 | 0.7786 | 0.8992 | 0.8324 |
| L35 | 0.9377 | 0.8096 | 0.9198 | 0.7618 | 0.9199 | 0.8579 |
| L40 | 0.9368 | 0.8055 | 0.9417 | 0.8113 | 0.9178 | 0.8673 |
| L45 | 0.9369 | 0.8238 | 0.9368 | 0.7783 | 0.9218 | 0.8799 |

## Multi-layer concat (mean, 13 layers)

| Train set | → cyber_3 test | → refusal test |
|---|---:|---:|
| cyber_3 only | 0.6476 | 0.7412 |
| refusal only | 0.8624 | 0.9137 |
| combined | 0.8826 | 0.9005 |
