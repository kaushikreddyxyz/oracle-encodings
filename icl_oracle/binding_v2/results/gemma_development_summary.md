| Model | Task | Shots | Classes | Mode | Layer | Beta | Seen | Seeds | Queries | Accuracy ± SE | P(correct) | Label mass | Full accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| google/gemma-2-2b | codebook | 128 | 8 | inject | 8 | 0.25 | True | 4 | 128 | 0.1250 ± 0.0000 | 0.1290 | 0.5950 | 0.1328 |
| google/gemma-2-2b | codebook | 128 | 8 | inject | 8 | 0.5 | True | 4 | 128 | 0.1328 ± 0.0078 | 0.1334 | 0.5980 | 0.1328 |
| google/gemma-2-2b | codebook | 128 | 8 | inject | 8 | 1.0 | True | 4 | 128 | 0.1484 ± 0.0150 | 0.1789 | 0.6151 | 0.1406 |
| google/gemma-2-2b | codebook | 128 | 8 | inject | 8 | 2.0 | True | 4 | 128 | 0.9453 ± 0.0547 | 0.8323 | 0.7883 | 0.9219 |
| google/gemma-2-2b | codebook | 128 | 8 | none | 8 | 0.0 | True | 4 | 128 | 0.1250 ± 0.0000 | 0.1273 | 0.5917 | 0.1250 |
| google/gemma-2-2b | codebook | 128 | 8 | token | 8 | 0.0 | True | 4 | 128 | 1.0000 ± 0.0000 | 1.0000 | 0.9940 | 1.0000 |
| google/gemma-2-2b | parity | 128 | 8 | inject | 8 | 0.25 | False | 4 | 128 | 0.1250 ± 0.0423 | 0.1475 | 0.6418 | 0.1250 |
| google/gemma-2-2b | parity | 128 | 8 | inject | 8 | 0.25 | True | 4 | 128 | 0.1562 ± 0.0383 | 0.1205 | 0.6436 | 0.1562 |
| google/gemma-2-2b | parity | 128 | 8 | inject | 8 | 0.5 | False | 4 | 128 | 0.1094 ± 0.0486 | 0.1474 | 0.6416 | 0.1094 |
| google/gemma-2-2b | parity | 128 | 8 | inject | 8 | 0.5 | True | 4 | 128 | 0.1562 ± 0.0383 | 0.1207 | 0.6428 | 0.1562 |
| google/gemma-2-2b | parity | 128 | 8 | inject | 8 | 1.0 | False | 4 | 128 | 0.1172 ± 0.0449 | 0.1476 | 0.6315 | 0.1172 |
| google/gemma-2-2b | parity | 128 | 8 | inject | 8 | 1.0 | True | 4 | 128 | 0.1484 ± 0.0391 | 0.1214 | 0.6306 | 0.1484 |
| google/gemma-2-2b | parity | 128 | 8 | inject | 8 | 2.0 | False | 4 | 128 | 0.1406 ± 0.0372 | 0.1450 | 0.5884 | 0.1406 |
| google/gemma-2-2b | parity | 128 | 8 | inject | 8 | 2.0 | True | 4 | 128 | 0.1406 ± 0.0393 | 0.1320 | 0.5833 | 0.1406 |
| google/gemma-2-2b | parity | 128 | 8 | none | 8 | 0.0 | False | 4 | 128 | 0.1328 ± 0.0391 | 0.1479 | 0.6409 | 0.1328 |
| google/gemma-2-2b | parity | 128 | 8 | none | 8 | 0.0 | True | 4 | 128 | 0.1641 ± 0.0391 | 0.1206 | 0.6431 | 0.1641 |
| google/gemma-2-2b | parity | 128 | 8 | token | 8 | 0.0 | False | 4 | 128 | 0.1250 ± 0.0338 | 0.1355 | 0.6411 | 0.1250 |
| google/gemma-2-2b | parity | 128 | 8 | token | 8 | 0.0 | True | 4 | 128 | 0.1797 ± 0.0322 | 0.1377 | 0.6411 | 0.1797 |
| google/gemma-2-2b | selection | 128 | 8 | inject | 8 | 0.25 | False | 4 | 128 | 0.1172 ± 0.0150 | 0.1199 | 0.6541 | 0.1172 |
| google/gemma-2-2b | selection | 128 | 8 | inject | 8 | 0.25 | True | 4 | 128 | 0.1484 ± 0.0322 | 0.1487 | 0.6566 | 0.1484 |
| google/gemma-2-2b | selection | 128 | 8 | inject | 8 | 0.5 | False | 4 | 128 | 0.1250 ± 0.0221 | 0.1211 | 0.6519 | 0.1250 |
| google/gemma-2-2b | selection | 128 | 8 | inject | 8 | 0.5 | True | 4 | 128 | 0.1484 ± 0.0322 | 0.1504 | 0.6544 | 0.1484 |
| google/gemma-2-2b | selection | 128 | 8 | inject | 8 | 1.0 | False | 4 | 128 | 0.1016 ± 0.0150 | 0.1261 | 0.6419 | 0.1016 |
| google/gemma-2-2b | selection | 128 | 8 | inject | 8 | 1.0 | True | 4 | 128 | 0.1484 ± 0.0322 | 0.1536 | 0.6454 | 0.1484 |
| google/gemma-2-2b | selection | 128 | 8 | inject | 8 | 2.0 | False | 4 | 128 | 0.1406 ± 0.0202 | 0.1611 | 0.6219 | 0.1328 |
| google/gemma-2-2b | selection | 128 | 8 | inject | 8 | 2.0 | True | 4 | 128 | 0.2109 ± 0.0369 | 0.1889 | 0.6288 | 0.2109 |
| google/gemma-2-2b | selection | 128 | 8 | none | 8 | 0.0 | False | 4 | 128 | 0.1094 ± 0.0202 | 0.1186 | 0.6538 | 0.1094 |
| google/gemma-2-2b | selection | 128 | 8 | none | 8 | 0.0 | True | 4 | 128 | 0.1641 ± 0.0197 | 0.1486 | 0.6561 | 0.1641 |
| google/gemma-2-2b | selection | 128 | 8 | token | 8 | 0.0 | False | 4 | 128 | 0.1562 ± 0.0556 | 0.1431 | 0.6618 | 0.1562 |
| google/gemma-2-2b | selection | 128 | 8 | token | 8 | 0.0 | True | 4 | 128 | 0.2812 ± 0.0675 | 0.1997 | 0.6649 | 0.2812 |

Matched clean-vector baselines on the same observed queries:

| Model/task/config | Seen | Method | Dev-selected setting | Accuracy ± SE |
| --- | --- | --- | --- | --- |
| google/gemma-2-2b / codebook / shots=128 / K=8 / inject / layer=8 / beta=0.25 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=8 / inject / layer=8 / beta=0.5 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=8 / inject / layer=8 / beta=1.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / codebook / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.25 | False | nearest_cosine | nearest_cosine | 0.2578 ± 0.0449 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.25 | False | cosine_vote | cosine_vote_t0.01 | 0.2422 ± 0.0267 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.25 | False | linear_ridge | linear_ridge_0.001 | 0.1016 ± 0.0296 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.25 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.5391 ± 0.0346 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.25 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.25 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.25 | True | linear_ridge | linear_ridge_0.001 | 0.3438 ± 0.0221 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.25 | True | quadratic_ridge | quadratic_ridge_0.001 | 0.9609 ± 0.0234 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.5 | False | nearest_cosine | nearest_cosine | 0.2578 ± 0.0449 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.5 | False | cosine_vote | cosine_vote_t0.01 | 0.2422 ± 0.0267 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.5 | False | linear_ridge | linear_ridge_0.001 | 0.1016 ± 0.0296 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.5 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.5391 ± 0.0346 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.5 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.5 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.5 | True | linear_ridge | linear_ridge_0.001 | 0.3438 ± 0.0221 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=0.5 | True | quadratic_ridge | quadratic_ridge_0.001 | 0.9609 ± 0.0234 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=1.0 | False | nearest_cosine | nearest_cosine | 0.2578 ± 0.0449 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=1.0 | False | cosine_vote | cosine_vote_t0.01 | 0.2422 ± 0.0267 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=1.0 | False | linear_ridge | linear_ridge_0.001 | 0.1016 ± 0.0296 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=1.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.5391 ± 0.0346 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=1.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=1.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=1.0 | True | linear_ridge | linear_ridge_0.001 | 0.3438 ± 0.0221 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=1.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 0.9609 ± 0.0234 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | nearest_cosine | nearest_cosine | 0.2578 ± 0.0449 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | cosine_vote | cosine_vote_t0.01 | 0.2422 ± 0.0267 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | linear_ridge | linear_ridge_0.001 | 0.1016 ± 0.0296 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.5391 ± 0.0346 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | linear_ridge | linear_ridge_0.001 | 0.3438 ± 0.0221 |
| google/gemma-2-2b / parity / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 0.9609 ± 0.0234 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.2578 ± 0.0449 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.2422 ± 0.0267 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.1016 ± 0.0296 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.5391 ± 0.0346 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | linear_ridge | linear_ridge_0.001 | 0.3438 ± 0.0221 |
| google/gemma-2-2b / parity / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 0.9609 ± 0.0234 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.2578 ± 0.0449 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.2422 ± 0.0267 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.1016 ± 0.0296 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 0.5391 ± 0.0346 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | linear_ridge | linear_ridge_0.001 | 0.3438 ± 0.0221 |
| google/gemma-2-2b / parity / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 0.9609 ± 0.0234 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.25 | False | nearest_cosine | nearest_cosine | 0.6797 ± 0.0369 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.25 | False | cosine_vote | cosine_vote_t0.01 | 0.9609 ± 0.0197 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.25 | False | linear_ridge | linear_ridge_0.001 | 0.9766 ± 0.0150 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.25 | False | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.25 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.25 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.25 | True | linear_ridge | linear_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.25 | True | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.5 | False | nearest_cosine | nearest_cosine | 0.6797 ± 0.0369 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.5 | False | cosine_vote | cosine_vote_t0.01 | 0.9609 ± 0.0197 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.5 | False | linear_ridge | linear_ridge_0.001 | 0.9766 ± 0.0150 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.5 | False | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.5 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.5 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.5 | True | linear_ridge | linear_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=0.5 | True | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=1.0 | False | nearest_cosine | nearest_cosine | 0.6797 ± 0.0369 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=1.0 | False | cosine_vote | cosine_vote_t0.01 | 0.9609 ± 0.0197 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=1.0 | False | linear_ridge | linear_ridge_0.001 | 0.9766 ± 0.0150 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=1.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=1.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=1.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=1.0 | True | linear_ridge | linear_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=1.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | nearest_cosine | nearest_cosine | 0.6797 ± 0.0369 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | cosine_vote | cosine_vote_t0.01 | 0.9609 ± 0.0197 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | linear_ridge | linear_ridge_0.001 | 0.9766 ± 0.0150 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | linear_ridge | linear_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / inject / layer=8 / beta=2.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.6797 ± 0.0369 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.9609 ± 0.0197 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.9766 ± 0.0150 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | linear_ridge | linear_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / none / layer=8 / beta=0.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | nearest_cosine | nearest_cosine | 0.6797 ± 0.0369 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | cosine_vote | cosine_vote_t0.01 | 0.9609 ± 0.0197 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | linear_ridge | linear_ridge_0.001 | 0.9766 ± 0.0150 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | False | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | nearest_cosine | nearest_cosine | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | cosine_vote | cosine_vote_t0.01 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | linear_ridge | linear_ridge_0.001 | 1.0000 ± 0.0000 |
| google/gemma-2-2b / selection / shots=128 / K=8 / token / layer=8 / beta=0.0 | True | quadratic_ridge | quadratic_ridge_0.001 | 1.0000 ± 0.0000 |
