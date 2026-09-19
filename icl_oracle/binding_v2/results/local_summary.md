| Model | Task | Shots | Classes | Mode | Layer | Beta | Seen | Seeds | Queries | Accuracy ± SE | P(correct) | Label mass | Full accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-0.6B-Base | codebook | 64 | 8 | inject | 4 | 1.0 | True | 2 | 32 | 0.3438 ± 0.0937 | 0.3275 | 0.4164 | 0.3438 |
| Qwen/Qwen3-0.6B-Base | codebook | 64 | 8 | inject | 4 | 2.0 | True | 2 | 32 | 0.1250 ± 0.0000 | 0.1353 | 0.3934 | 0.1250 |
| Qwen/Qwen3-0.6B-Base | codebook | 64 | 8 | inject | 8 | 0.25 | True | 2 | 32 | 0.5312 ± 0.0937 | 0.4738 | 0.4457 | 0.4062 |
| Qwen/Qwen3-0.6B-Base | codebook | 64 | 8 | inject | 8 | 0.5 | True | 2 | 32 | 0.5000 ± 0.1250 | 0.4616 | 0.6194 | 0.5000 |
| Qwen/Qwen3-0.6B-Base | codebook | 64 | 8 | inject | 8 | 1.0 | True | 2 | 32 | 0.1562 ± 0.0937 | 0.2426 | 0.3032 | 0.1562 |
| Qwen/Qwen3-0.6B-Base | codebook | 64 | 8 | inject | 8 | 2.0 | True | 2 | 32 | 0.0938 ± 0.0312 | 0.1076 | 0.3790 | 0.0625 |
| Qwen/Qwen3-0.6B-Base | codebook | 64 | 8 | none | 4 | 0.0 | True | 2 | 32 | 0.1250 ± 0.0000 | 0.1312 | 0.3766 | 0.1250 |
| Qwen/Qwen3-0.6B-Base | codebook | 64 | 8 | token | 4 | 0.0 | True | 2 | 32 | 1.0000 ± 0.0000 | 1.0000 | 0.9826 | 1.0000 |
| Qwen/Qwen3-0.6B-Base | parity | 64 | 8 | inject | 4 | 1.0 | False | 2 | 32 | 0.1562 ± 0.0312 | 0.1358 | 0.1933 | 0.0625 |
| Qwen/Qwen3-0.6B-Base | parity | 64 | 8 | inject | 4 | 2.0 | False | 2 | 32 | 0.1250 ± 0.0000 | 0.1333 | 0.2482 | 0.0625 |
| Qwen/Qwen3-0.6B-Base | parity | 64 | 8 | none | 4 | 0.0 | False | 2 | 32 | 0.1875 ± 0.0625 | 0.1505 | 0.2906 | 0.1250 |
| Qwen/Qwen3-0.6B-Base | parity | 64 | 8 | token | 4 | 0.0 | False | 2 | 32 | 0.1875 ± 0.0000 | 0.1545 | 0.2526 | 0.0938 |
| Qwen/Qwen3-0.6B-Base | parity | 128 | 8 | token | 8 | 0.0 | False | 2 | 32 | 0.0625 ± 0.0000 | 0.0820 | 0.7422 | 0.0625 |
| Qwen/Qwen3-0.6B-Base | selection | 64 | 8 | inject | 4 | 1.0 | False | 2 | 32 | 0.1562 ± 0.0312 | 0.1566 | 0.4204 | 0.1562 |
| Qwen/Qwen3-0.6B-Base | selection | 64 | 8 | inject | 4 | 2.0 | False | 2 | 32 | 0.1250 ± 0.0625 | 0.1306 | 0.3784 | 0.1250 |
| Qwen/Qwen3-0.6B-Base | selection | 64 | 8 | none | 4 | 0.0 | False | 2 | 32 | 0.1562 ± 0.0937 | 0.1301 | 0.3990 | 0.1250 |
| Qwen/Qwen3-0.6B-Base | selection | 64 | 8 | token | 4 | 0.0 | False | 2 | 32 | 0.1562 ± 0.0937 | 0.1494 | 0.3520 | 0.1562 |
| Qwen/Qwen3-0.6B-Base | selection | 128 | 8 | token | 8 | 0.0 | False | 2 | 32 | 0.1250 ± 0.0625 | 0.1259 | 0.6985 | 0.1250 |

Matched clean-vector baselines on the same observed queries:

| Model/task/config | Seen | Method | Dev-selected setting | Accuracy ± SE |
| --- | --- | --- | --- | --- |
| Qwen/Qwen3-0.6B-Base / codebook / shots=64 / K=8 / inject / layer=4 / beta=1.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / codebook / shots=64 / K=8 / inject / layer=4 / beta=2.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / codebook / shots=64 / K=8 / inject / layer=8 / beta=0.25 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / codebook / shots=64 / K=8 / inject / layer=8 / beta=0.5 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / codebook / shots=64 / K=8 / inject / layer=8 / beta=1.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / codebook / shots=64 / K=8 / inject / layer=8 / beta=2.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / codebook / shots=64 / K=8 / none / layer=4 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / codebook / shots=64 / K=8 / token / layer=4 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / inject / layer=4 / beta=1.0 | False | nearest_cosine | nearest_cosine | 0.3750 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / inject / layer=4 / beta=1.0 | False | cosine_vote | cosine_vote_t0.01 | 0.2500 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / inject / layer=4 / beta=1.0 | False | linear_ridge | linear_ridge_1 | 0.1562 ± 0.0312 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / inject / layer=4 / beta=1.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.4375 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / inject / layer=4 / beta=2.0 | False | nearest_cosine | nearest_cosine | 0.3750 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / inject / layer=4 / beta=2.0 | False | cosine_vote | cosine_vote_t0.01 | 0.2500 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / inject / layer=4 / beta=2.0 | False | linear_ridge | linear_ridge_1 | 0.1562 ± 0.0312 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / inject / layer=4 / beta=2.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.4375 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / none / layer=4 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.3750 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / none / layer=4 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.2500 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / none / layer=4 / beta=0.0 | False | linear_ridge | linear_ridge_1 | 0.1562 ± 0.0312 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / none / layer=4 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.4375 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / token / layer=4 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.3750 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / token / layer=4 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.2500 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / token / layer=4 / beta=0.0 | False | linear_ridge | linear_ridge_1 | 0.1562 ± 0.0312 |
| Qwen/Qwen3-0.6B-Base / parity / shots=64 / K=8 / token / layer=4 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.4375 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.2188 ± 0.0312 |
| Qwen/Qwen3-0.6B-Base / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.3438 ± 0.0312 |
| Qwen/Qwen3-0.6B-Base / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.1562 ± 0.0312 |
| Qwen/Qwen3-0.6B-Base / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.5312 ± 0.0312 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / inject / layer=4 / beta=1.0 | False | nearest_cosine | nearest_cosine | 0.6875 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / inject / layer=4 / beta=1.0 | False | cosine_vote | cosine_vote_t0.03 | 0.8125 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / inject / layer=4 / beta=1.0 | False | linear_ridge | linear_ridge_0.001 | 0.8438 ± 0.0312 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / inject / layer=4 / beta=1.0 | False | quadratic_ridge | quadratic_ridge_0.1 | 1.0000 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / inject / layer=4 / beta=2.0 | False | nearest_cosine | nearest_cosine | 0.6875 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / inject / layer=4 / beta=2.0 | False | cosine_vote | cosine_vote_t0.03 | 0.8125 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / inject / layer=4 / beta=2.0 | False | linear_ridge | linear_ridge_0.001 | 0.8438 ± 0.0312 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / inject / layer=4 / beta=2.0 | False | quadratic_ridge | quadratic_ridge_0.1 | 1.0000 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / none / layer=4 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.6875 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / none / layer=4 / beta=0.0 | False | cosine_vote | cosine_vote_t0.03 | 0.8125 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / none / layer=4 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.8438 ± 0.0312 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / none / layer=4 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.1 | 1.0000 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / token / layer=4 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.6875 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / token / layer=4 / beta=0.0 | False | cosine_vote | cosine_vote_t0.03 | 0.8125 ± 0.0625 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / token / layer=4 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.8438 ± 0.0312 |
| Qwen/Qwen3-0.6B-Base / selection / shots=64 / K=8 / token / layer=4 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.1 | 1.0000 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.5625 ± 0.0000 |
| Qwen/Qwen3-0.6B-Base / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.9062 ± 0.0937 |
| Qwen/Qwen3-0.6B-Base / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.9688 ± 0.0312 |
| Qwen/Qwen3-0.6B-Base / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
