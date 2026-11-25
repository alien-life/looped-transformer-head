# Per-loop decoding probe

## frozen backbone

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 0 | answer | 0.126 |  |  |  |  |
| 2 | 203 | 0 | answer | 0.010 | 0.015 | 0.044 |  | 0.941 |
| 3 | 206 | 0 | answer | 0.000 | 0.019 | 0.000 | 0.005 | 0.976 |

## 1x-plain (R=1, final answer only)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.969 |  |  |  |  |
| 2 | 203 | 1 | answer | 0.335 | 0.000 | 0.335 |  | 0.665 |
| 3 | 206 | 1 | answer | 0.184 | 0.000 | 0.000 | 0.184 | 0.816 |

## 3x-chain (R=3, chain-supervised)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.969 |  |  |  |  |
| 1 | 191 | 2 | answer | 0.974 |  |  |  |  |
| 1 | 191 | 3 | answer | 0.974 |  |  |  |  |
| 2 | 203 | 1 | hop 1 | 0.916 | 0.916 | 0.000 |  | 0.084 |
| 2 | 203 | 2 | answer | 0.404 | 0.000 | 0.404 |  | 0.596 |
| 2 | 203 | 3 | answer | 0.424 | 0.000 | 0.424 |  | 0.576 |
| 3 | 206 | 1 | hop 1 | 0.908 | 0.908 | 0.000 | 0.000 | 0.092 |
| 3 | 206 | 2 | hop 2 | 0.374 | 0.000 | 0.374 | 0.000 | 0.626 |
| 3 | 206 | 3 | answer | 0.233 | 0.000 | 0.010 | 0.233 | 0.757 |

`holds hop j`: share of examples whose top prediction at the entity token is hop j's entity (`other`: anything else). Loop 0 is the frozen backbone.
