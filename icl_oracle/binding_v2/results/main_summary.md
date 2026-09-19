| Model | Task | Shots | Classes | Mode | Layer | Beta | Seen | Seeds | Queries | Accuracy ± SE | P(correct) | Label mass | Full accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-7B | codebook | 128 | 4 | inject | 4 | 1.0 | True | 8 | 256 | 0.9922 ± 0.0051 | 0.9897 | 0.9950 | 0.9922 |
| Qwen/Qwen2.5-7B | codebook | 128 | 4 | none | 4 | 0.0 | True | 8 | 256 | 0.2539 ± 0.0191 | 0.2518 | 0.8345 | 0.2500 |
| Qwen/Qwen2.5-7B | codebook | 128 | 4 | token | 4 | 0.0 | True | 8 | 256 | 1.0000 ± 0.0000 | 1.0000 | 0.9992 | 1.0000 |
| Qwen/Qwen2.5-7B | codebook | 128 | 8 | inject | 4 | 1.0 | True | 8 | 256 | 0.9844 ± 0.0156 | 0.9817 | 0.9692 | 0.9844 |
| Qwen/Qwen2.5-7B | codebook | 128 | 8 | none | 4 | 0.0 | True | 8 | 256 | 0.1172 ± 0.0078 | 0.1264 | 0.6619 | 0.1133 |
| Qwen/Qwen2.5-7B | codebook | 128 | 8 | token | 4 | 0.0 | True | 8 | 256 | 0.9844 ± 0.0156 | 0.9844 | 0.9785 | 0.9844 |
| Qwen/Qwen2.5-7B | codebook | 128 | 16 | inject | 4 | 1.0 | True | 8 | 256 | 0.9727 ± 0.0109 | 0.9274 | 0.9292 | 0.9648 |
| Qwen/Qwen2.5-7B | codebook | 128 | 16 | none | 4 | 0.0 | True | 8 | 256 | 0.0664 ± 0.0092 | 0.0630 | 0.4777 | 0.0625 |
| Qwen/Qwen2.5-7B | codebook | 128 | 16 | token | 4 | 0.0 | True | 8 | 256 | 1.0000 ± 0.0000 | 1.0000 | 0.9890 | 0.9922 |
| Qwen/Qwen2.5-7B | codebook | 128 | 32 | inject | 4 | 1.0 | True | 8 | 256 | 0.6797 ± 0.0456 | 0.5992 | 0.6152 | 0.6641 |
| Qwen/Qwen2.5-7B | codebook | 128 | 32 | none | 4 | 0.0 | True | 8 | 256 | 0.0312 ± 0.0059 | 0.0312 | 0.2098 | 0.0195 |
| Qwen/Qwen2.5-7B | codebook | 128 | 32 | token | 4 | 0.0 | True | 8 | 256 | 1.0000 ± 0.0000 | 1.0000 | 0.9928 | 0.9961 |
| Qwen/Qwen2.5-7B | parity | 128 | 8 | inject | 4 | 2.0 | False | 8 | 256 | 0.1445 ± 0.0212 | 0.1231 | 0.5958 | 0.1445 |
| Qwen/Qwen2.5-7B | parity | 128 | 8 | inject | 4 | 2.0 | True | 8 | 256 | 0.2266 ± 0.0282 | 0.1706 | 0.5963 | 0.2305 |
| Qwen/Qwen2.5-7B | parity | 128 | 8 | none | 4 | 0.0 | False | 8 | 256 | 0.0977 ± 0.0224 | 0.1163 | 0.6061 | 0.0977 |
| Qwen/Qwen2.5-7B | parity | 128 | 8 | none | 4 | 0.0 | True | 8 | 256 | 0.1328 ± 0.0098 | 0.1268 | 0.6049 | 0.1211 |
| Qwen/Qwen2.5-7B | parity | 128 | 8 | token | 4 | 0.0 | False | 8 | 256 | 0.0859 ± 0.0194 | 0.1166 | 0.6442 | 0.0781 |
| Qwen/Qwen2.5-7B | parity | 128 | 8 | token | 4 | 0.0 | True | 8 | 256 | 0.1719 ± 0.0196 | 0.1502 | 0.6408 | 0.1484 |
| Qwen/Qwen2.5-7B | selection | 128 | 8 | inject | 4 | 1.0 | False | 8 | 256 | 0.3086 ± 0.0444 | 0.2553 | 0.6926 | 0.3164 |
| Qwen/Qwen2.5-7B | selection | 128 | 8 | inject | 4 | 1.0 | True | 8 | 256 | 0.4766 ± 0.0386 | 0.3531 | 0.7026 | 0.4844 |
| Qwen/Qwen2.5-7B | selection | 128 | 8 | none | 4 | 0.0 | False | 8 | 256 | 0.1172 ± 0.0164 | 0.1182 | 0.6420 | 0.1172 |
| Qwen/Qwen2.5-7B | selection | 128 | 8 | none | 4 | 0.0 | True | 8 | 256 | 0.1250 ± 0.0156 | 0.1224 | 0.6389 | 0.1250 |
| Qwen/Qwen2.5-7B | selection | 128 | 8 | token | 4 | 0.0 | False | 8 | 256 | 0.2852 ± 0.0467 | 0.2259 | 0.7117 | 0.2891 |
| Qwen/Qwen2.5-7B | selection | 128 | 8 | token | 4 | 0.0 | True | 8 | 256 | 0.4062 ± 0.0491 | 0.3118 | 0.7267 | 0.4062 |
| google/gemma-2-2b | codebook | 128 | 4 | inject | 8 | 2.0 | True | 8 | 256 | 0.9570 ± 0.0220 | 0.9388 | 0.9469 | 0.9570 |
| google/gemma-2-2b | codebook | 128 | 4 | none | 8 | 0.0 | True | 8 | 256 | 0.2461 ± 0.0039 | 0.2484 | 0.7797 | 0.2344 |
| google/gemma-2-2b | codebook | 128 | 4 | token | 8 | 0.0 | True | 8 | 256 | 1.0000 ± 0.0000 | 0.9998 | 0.9927 | 1.0000 |
| google/gemma-2-2b | codebook | 128 | 8 | inject | 8 | 2.0 | True | 8 | 256 | 0.9492 ± 0.0250 | 0.8104 | 0.7010 | 0.9219 |
| google/gemma-2-2b | codebook | 128 | 8 | none | 8 | 0.0 | True | 8 | 256 | 0.1289 ± 0.0039 | 0.1222 | 0.5133 | 0.1172 |
| google/gemma-2-2b | codebook | 128 | 8 | token | 8 | 0.0 | True | 8 | 256 | 1.0000 ± 0.0000 | 0.9999 | 0.9740 | 0.9844 |
| google/gemma-2-2b | codebook | 128 | 16 | inject | 8 | 2.0 | True | 8 | 256 | 0.5703 ± 0.0635 | 0.4798 | 0.3750 | 0.5391 |
| google/gemma-2-2b | codebook | 128 | 16 | none | 8 | 0.0 | True | 8 | 256 | 0.0664 ± 0.0071 | 0.0613 | 0.2423 | 0.0547 |
| google/gemma-2-2b | codebook | 128 | 16 | token | 8 | 0.0 | True | 8 | 256 | 0.9961 ± 0.0039 | 0.9977 | 0.9826 | 0.9961 |
| google/gemma-2-2b | codebook | 128 | 32 | inject | 8 | 2.0 | True | 8 | 256 | 0.2383 ± 0.0569 | 0.1860 | 0.1266 | 0.1992 |
| google/gemma-2-2b | codebook | 128 | 32 | none | 8 | 0.0 | True | 8 | 256 | 0.0273 ± 0.0071 | 0.0288 | 0.0893 | 0.0156 |
| google/gemma-2-2b | codebook | 128 | 32 | token | 8 | 0.0 | True | 8 | 256 | 0.9922 ± 0.0051 | 0.9889 | 0.9440 | 0.9805 |
| google/gemma-2-2b | parity | 128 | 8 | inject | 8 | 2.0 | False | 8 | 256 | 0.1094 ± 0.0132 | 0.1149 | 0.6177 | 0.1094 |
| google/gemma-2-2b | parity | 128 | 8 | inject | 8 | 2.0 | True | 8 | 256 | 0.1562 ± 0.0177 | 0.1479 | 0.6180 | 0.1602 |
| google/gemma-2-2b | parity | 128 | 8 | none | 8 | 0.0 | False | 8 | 256 | 0.1172 ± 0.0078 | 0.1193 | 0.6654 | 0.1211 |
| google/gemma-2-2b | parity | 128 | 8 | none | 8 | 0.0 | True | 8 | 256 | 0.1328 ± 0.0194 | 0.1350 | 0.6649 | 0.1250 |
| google/gemma-2-2b | parity | 128 | 8 | token | 8 | 0.0 | False | 8 | 256 | 0.1094 ± 0.0145 | 0.1177 | 0.6691 | 0.1094 |
| google/gemma-2-2b | parity | 128 | 8 | token | 8 | 0.0 | True | 8 | 256 | 0.1602 ± 0.0124 | 0.1414 | 0.6682 | 0.1758 |
| google/gemma-2-2b | selection | 128 | 8 | inject | 8 | 2.0 | False | 8 | 256 | 0.1445 ± 0.0156 | 0.1455 | 0.6019 | 0.1445 |
| google/gemma-2-2b | selection | 128 | 8 | inject | 8 | 2.0 | True | 8 | 256 | 0.2109 ± 0.0282 | 0.1834 | 0.6075 | 0.2070 |
| google/gemma-2-2b | selection | 128 | 8 | none | 8 | 0.0 | False | 8 | 256 | 0.1016 ± 0.0242 | 0.1111 | 0.6683 | 0.1016 |
| google/gemma-2-2b | selection | 128 | 8 | none | 8 | 0.0 | True | 8 | 256 | 0.1172 ± 0.0211 | 0.1216 | 0.6675 | 0.1172 |
| google/gemma-2-2b | selection | 128 | 8 | token | 8 | 0.0 | False | 8 | 256 | 0.1602 ± 0.0315 | 0.1349 | 0.6978 | 0.1523 |
| google/gemma-2-2b | selection | 128 | 8 | token | 8 | 0.0 | True | 8 | 256 | 0.2148 ± 0.0341 | 0.1703 | 0.6985 | 0.2109 |

Matched clean-vector baselines on the same observed queries:

| Model/task/config | Seen | Method | Dev-selected setting | Accuracy ± SE |
| --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-7B / codebook / shots=128 / K=4 / inject / layer=4 / beta=1.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / codebook / shots=128 / K=4 / none / layer=4 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / codebook / shots=128 / K=4 / token / layer=4 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / codebook / shots=128 / K=8 / inject / layer=4 / beta=1.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / codebook / shots=128 / K=8 / none / layer=4 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / codebook / shots=128 / K=8 / token / layer=4 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / codebook / shots=128 / K=16 / inject / layer=4 / beta=1.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / codebook / shots=128 / K=16 / none / layer=4 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / codebook / shots=128 / K=16 / token / layer=4 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / codebook / shots=128 / K=32 / inject / layer=4 / beta=1.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / codebook / shots=128 / K=32 / none / layer=4 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / codebook / shots=128 / K=32 / token / layer=4 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / inject / layer=4 / beta=2.0 | False | nearest_cosine | nearest_cosine | 0.2266 ± 0.0372 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / inject / layer=4 / beta=2.0 | False | cosine_vote | cosine_vote_t0.01 | 0.2539 ± 0.0273 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / inject / layer=4 / beta=2.0 | False | linear_ridge | linear_ridge_0.001 | 0.0898 ± 0.0172 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / inject / layer=4 / beta=2.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.5234 ± 0.0282 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / inject / layer=4 / beta=2.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / inject / layer=4 / beta=2.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / inject / layer=4 / beta=2.0 | True | linear_ridge | linear_ridge_0.001 | 0.3086 ± 0.0172 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / inject / layer=4 / beta=2.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 0.9336 ± 0.0172 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / none / layer=4 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.2266 ± 0.0372 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / none / layer=4 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.2539 ± 0.0273 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / none / layer=4 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.0898 ± 0.0172 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / none / layer=4 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.5234 ± 0.0282 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / none / layer=4 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / none / layer=4 / beta=0.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / none / layer=4 / beta=0.0 | True | linear_ridge | linear_ridge_0.001 | 0.3086 ± 0.0172 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / none / layer=4 / beta=0.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 0.9336 ± 0.0172 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / token / layer=4 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.2266 ± 0.0372 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / token / layer=4 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.2539 ± 0.0273 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / token / layer=4 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.0898 ± 0.0172 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / token / layer=4 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.5234 ± 0.0282 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / token / layer=4 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / token / layer=4 / beta=0.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / token / layer=4 / beta=0.0 | True | linear_ridge | linear_ridge_0.001 | 0.3086 ± 0.0172 |
| Qwen/Qwen2.5-7B / parity / shots=128 / K=8 / token / layer=4 / beta=0.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 0.9336 ± 0.0172 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / inject / layer=4 / beta=1.0 | False | nearest_cosine | nearest_cosine | 0.5938 ± 0.0264 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / inject / layer=4 / beta=1.0 | False | cosine_vote | cosine_vote_t0.01 | 0.9492 ± 0.0144 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / inject / layer=4 / beta=1.0 | False | linear_ridge | linear_ridge_0.001 | 0.9922 ± 0.0051 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / inject / layer=4 / beta=1.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / inject / layer=4 / beta=1.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / inject / layer=4 / beta=1.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / inject / layer=4 / beta=1.0 | True | linear_ridge | linear_ridge_0.001 | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / inject / layer=4 / beta=1.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / none / layer=4 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.5938 ± 0.0264 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / none / layer=4 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.9492 ± 0.0144 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / none / layer=4 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.9922 ± 0.0051 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / none / layer=4 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / none / layer=4 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / none / layer=4 / beta=0.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / none / layer=4 / beta=0.0 | True | linear_ridge | linear_ridge_0.001 | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / none / layer=4 / beta=0.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / token / layer=4 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.5938 ± 0.0264 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / token / layer=4 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.9492 ± 0.0144 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / token / layer=4 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.9922 ± 0.0051 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / token / layer=4 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / token / layer=4 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / token / layer=4 / beta=0.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / token / layer=4 / beta=0.0 | True | linear_ridge | linear_ridge_0.001 | 1.0000 ± 0.0000 |
| Qwen/Qwen2.5-7B / selection / shots=128 / K=8 / token / layer=4 / beta=0.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=4 / inject / layer=8 / beta=2.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=4 / none / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=4 / token / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=16 / inject / layer=8 / beta=2.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=16 / none / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=16 / token / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=32 / inject / layer=8 / beta=2.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=32 / none / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=32 / token / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | nearest_cosine | nearest_cosine | 0.2266 ± 0.0372 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | cosine_vote | cosine_vote_t0.01 | 0.2539 ± 0.0273 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | linear_ridge | linear_ridge_0.001 | 0.0898 ± 0.0172 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.5234 ± 0.0282 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | linear_ridge | linear_ridge_0.001 | 0.3086 ± 0.0172 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 0.9336 ± 0.0172 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.2266 ± 0.0372 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.2539 ± 0.0273 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.0898 ± 0.0172 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.5234 ± 0.0282 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | linear_ridge | linear_ridge_0.001 | 0.3086 ± 0.0172 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 0.9336 ± 0.0172 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.2266 ± 0.0372 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.2539 ± 0.0273 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.0898 ± 0.0172 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.5234 ± 0.0282 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | linear_ridge | linear_ridge_0.001 | 0.3086 ± 0.0172 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 0.9336 ± 0.0172 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | nearest_cosine | nearest_cosine | 0.5938 ± 0.0264 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | cosine_vote | cosine_vote_t0.01 | 0.9492 ± 0.0144 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | linear_ridge | linear_ridge_0.001 | 0.9922 ± 0.0051 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | linear_ridge | linear_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.5938 ± 0.0264 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.9492 ± 0.0144 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.9922 ± 0.0051 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | linear_ridge | linear_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.5938 ± 0.0264 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.9492 ± 0.0144 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.9922 ± 0.0051 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | linear_ridge | linear_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
