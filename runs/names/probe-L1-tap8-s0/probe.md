# Per-loop decoding probe

## frozen backbone

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 0 | answer | 0.126 |  |  |  |  |
| 2 | 203 | 0 | answer | 0.010 | 0.015 | 0.044 |  | 0.941 |
| 3 | 206 | 0 | answer | 0.000 | 0.019 | 0.000 | 0.005 | 0.976 |

## 1x-plain-L1-tap8-s0 (R=1, final answer only)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.586 |  |  |  |  |
| 2 | 203 | 1 | answer | 0.291 | 0.000 | 0.291 |  | 0.709 |
| 3 | 206 | 1 | answer | 0.180 | 0.000 | 0.000 | 0.180 | 0.820 |

## 3x-plain-L1-tap8-s0 (R=3, final answer only)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.492 |  |  |  |  |
| 1 | 191 | 2 | answer | 0.524 |  |  |  |  |
| 1 | 191 | 3 | answer | 0.592 |  |  |  |  |
| 2 | 203 | 1 | hop 1 | 0.015 | 0.015 | 0.251 |  | 0.734 |
| 2 | 203 | 2 | answer | 0.296 | 0.000 | 0.296 |  | 0.704 |
| 2 | 203 | 3 | answer | 0.305 | 0.000 | 0.305 |  | 0.695 |
| 3 | 206 | 1 | hop 1 | 0.053 | 0.053 | 0.044 | 0.039 | 0.864 |
| 3 | 206 | 2 | hop 2 | 0.005 | 0.000 | 0.005 | 0.184 | 0.811 |
| 3 | 206 | 3 | answer | 0.214 | 0.000 | 0.000 | 0.214 | 0.786 |

## 2x-chain-L1-tap8-s0 (R=2, chain-supervised)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.597 |  |  |  |  |
| 1 | 191 | 2 | answer | 0.586 |  |  |  |  |
| 2 | 203 | 1 | hop 1 | 0.315 | 0.315 | 0.000 |  | 0.685 |
| 2 | 203 | 2 | answer | 0.276 | 0.000 | 0.276 |  | 0.724 |
| 3 | 206 | 1 | hop 1 | 0.456 | 0.456 | 0.000 | 0.000 | 0.544 |
| 3 | 206 | 2 | answer | 0.189 | 0.000 | 0.000 | 0.189 | 0.811 |

## 3x-chain-L1-tap8-s0 (R=3, chain-supervised)

| hops | n | loop | trained target | on target | holds hop 1 | holds hop 2 | holds hop 3 | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 191 | 1 | answer | 0.592 |  |  |  |  |
| 1 | 191 | 2 | answer | 0.592 |  |  |  |  |
| 1 | 191 | 3 | answer | 0.586 |  |  |  |  |
| 2 | 203 | 1 | hop 1 | 0.345 | 0.345 | 0.000 |  | 0.655 |
| 2 | 203 | 2 | answer | 0.271 | 0.000 | 0.271 |  | 0.729 |
| 2 | 203 | 3 | answer | 0.271 | 0.000 | 0.271 |  | 0.729 |
| 3 | 206 | 1 | hop 1 | 0.442 | 0.442 | 0.000 | 0.000 | 0.558 |
| 3 | 206 | 2 | hop 2 | 0.291 | 0.000 | 0.291 | 0.000 | 0.709 |
| 3 | 206 | 3 | answer | 0.194 | 0.000 | 0.000 | 0.194 | 0.806 |

`holds hop j`: share of examples whose top prediction at the entity token is hop j's entity (`other`: anything else). Loop 0 is the frozen backbone.
