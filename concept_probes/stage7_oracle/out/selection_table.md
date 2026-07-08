# Stage 7-Oracle Phase 0 — probe selection audit table

Chosen layers: [6, 8, 14]  |  Ablation layer: 8  |  AUROC threshold: 0.9  |  relaxed fallback used: False

## Per-layer mean best-arm example-AUROC (all 64 concepts)

| layer | mean best-arm AUROC | chosen |
|---|---|---|
| 1 | 0.9569 |  |
| 3 | 0.9573 |  |
| 6 | 0.9629 | YES |
| 8 | 0.9645 | YES |
| 10 | 0.9613 |  |
| 12 | 0.9606 |  |
| 14 | 0.9601 | YES |
| 16 | 0.9543 |  |
| 18 | 0.9528 |  |
| 20 | 0.9491 |  |
| 23 | 0.9446 |  |
| 25 | 0.9194 |  |

## Ablation-layer vote (e5_salient_layer_corrected over survivors)

histogram: {6: 1, 8: 22, 10: 4, 12: 17, 14: 4, 16: 4, 18: 2}  ->  mode = 8

## Full audit: every concept x chosen-layer x arm

| family | concept | layer | arm | auroc | token_rho | n_pos_ex | n_neg_ex | best_arm | pass(>=0.90) |
|---|---|---|---|---|---|---|---|---|---|
| color_wheel | blue | 6 | ridge | 0.9498 | 0.1093 | 147 | 1209 | <== | PASS |
| color_wheel | blue | 6 | lda | 0.9071 | 0.1048 | 147 | 1209 |  |  |
| color_wheel | blue | 6 | dom | 0.8546 | 0.1129 | 147 | 1209 |  |  |
| color_wheel | blue | 6 | logistic | 0.6192 | 0.0546 | 147 | 1209 |  |  |
| color_wheel | blue | 8 | ridge | 0.9508 | 0.1113 | 147 | 1209 | <== | PASS |
| color_wheel | blue | 8 | lda | 0.9028 | 0.1050 | 147 | 1209 |  |  |
| color_wheel | blue | 8 | dom | 0.8195 | 0.1102 | 147 | 1209 |  |  |
| color_wheel | blue | 8 | logistic | 0.5617 | 0.0612 | 147 | 1209 |  |  |
| color_wheel | blue | 14 | ridge | 0.9376 | 0.1117 | 147 | 1209 | <== | PASS |
| color_wheel | blue | 14 | lda | 0.9001 | 0.1076 | 147 | 1209 |  |  |
| color_wheel | blue | 14 | dom | 0.8508 | 0.1111 | 147 | 1209 |  |  |
| color_wheel | blue | 14 | logistic | 0.6541 | 0.0682 | 147 | 1209 |  |  |
| color_wheel | blue-green | 6 | dom | 0.9833 | 0.0765 | 38 | 1318 | <== | PASS |
| color_wheel | blue-green | 6 | lda | 0.9634 | 0.0744 | 38 | 1318 |  |  |
| color_wheel | blue-green | 6 | ridge | 0.9601 | 0.0729 | 38 | 1318 |  |  |
| color_wheel | blue-green | 6 | logistic | 0.6406 | 0.0500 | 38 | 1318 |  |  |
| color_wheel | blue-green | 8 | lda | 0.9478 | 0.0754 | 38 | 1318 | <== | PASS |
| color_wheel | blue-green | 8 | dom | 0.9405 | 0.0755 | 38 | 1318 |  |  |
| color_wheel | blue-green | 8 | ridge | 0.9398 | 0.0748 | 38 | 1318 |  |  |
| color_wheel | blue-green | 8 | logistic | 0.6087 | 0.0517 | 38 | 1318 |  |  |
| color_wheel | blue-green | 14 | lda | 0.9543 | 0.0765 | 38 | 1318 | <== | PASS |
| color_wheel | blue-green | 14 | ridge | 0.9477 | 0.0759 | 38 | 1318 |  |  |
| color_wheel | blue-green | 14 | dom | 0.9435 | 0.0754 | 38 | 1318 |  |  |
| color_wheel | blue-green | 14 | logistic | 0.6144 | 0.0554 | 38 | 1318 |  |  |
| color_wheel | blue-violet | 6 | lda | 0.9131 | 0.0440 | 15 | 1341 | <== | PASS |
| color_wheel | blue-violet | 6 | dom | 0.9046 | 0.0490 | 15 | 1341 |  |  |
| color_wheel | blue-violet | 6 | ridge | 0.8974 | 0.0440 | 15 | 1341 |  |  |
| color_wheel | blue-violet | 6 | logistic | 0.6690 | 0.0114 | 15 | 1341 |  |  |
| color_wheel | blue-violet | 8 | dom | 0.8895 | 0.0484 | 15 | 1341 | <== |  |
| color_wheel | blue-violet | 8 | lda | 0.8813 | 0.0434 | 15 | 1341 |  |  |
| color_wheel | blue-violet | 8 | ridge | 0.8765 | 0.0462 | 15 | 1341 |  |  |
| color_wheel | blue-violet | 8 | logistic | 0.5804 | 0.0088 | 15 | 1341 |  |  |
| color_wheel | blue-violet | 14 | lda | 0.9253 | 0.0478 | 15 | 1341 | <== | PASS |
| color_wheel | blue-violet | 14 | ridge | 0.9167 | 0.0485 | 15 | 1341 |  |  |
| color_wheel | blue-violet | 14 | dom | 0.9043 | 0.0487 | 15 | 1341 |  |  |
| color_wheel | blue-violet | 14 | logistic | 0.7050 | 0.0326 | 15 | 1341 |  |  |
| color_wheel | green | 6 | ridge | 0.9485 | 0.1071 | 130 | 1226 | <== | PASS |
| color_wheel | green | 6 | lda | 0.9228 | 0.1036 | 130 | 1226 |  |  |
| color_wheel | green | 6 | dom | 0.8827 | 0.1169 | 130 | 1226 |  |  |
| color_wheel | green | 6 | logistic | 0.6606 | 0.0758 | 130 | 1226 |  |  |
| color_wheel | green | 8 | ridge | 0.9649 | 0.1066 | 130 | 1226 | <== | PASS |
| color_wheel | green | 8 | lda | 0.9533 | 0.1036 | 130 | 1226 |  |  |
| color_wheel | green | 8 | dom | 0.7961 | 0.1139 | 130 | 1226 |  |  |
| color_wheel | green | 8 | logistic | 0.6704 | 0.0614 | 130 | 1226 |  |  |
| color_wheel | green | 14 | ridge | 0.9607 | 0.1110 | 130 | 1226 | <== | PASS |
| color_wheel | green | 14 | lda | 0.9380 | 0.1100 | 130 | 1226 |  |  |
| color_wheel | green | 14 | dom | 0.8280 | 0.1150 | 130 | 1226 |  |  |
| color_wheel | green | 14 | logistic | 0.6611 | 0.0736 | 130 | 1226 |  |  |
| color_wheel | orange | 6 | ridge | 0.9861 | 0.0830 | 78 | 1278 | <== | PASS |
| color_wheel | orange | 6 | lda | 0.9718 | 0.0753 | 78 | 1278 |  |  |
| color_wheel | orange | 6 | dom | 0.9453 | 0.0862 | 78 | 1278 |  |  |
| color_wheel | orange | 6 | logistic | 0.6260 | 0.0400 | 78 | 1278 |  |  |
| color_wheel | orange | 8 | ridge | 0.9908 | 0.0817 | 78 | 1278 | <== | PASS |
| color_wheel | orange | 8 | lda | 0.9849 | 0.0761 | 78 | 1278 |  |  |
| color_wheel | orange | 8 | dom | 0.9006 | 0.0859 | 78 | 1278 |  |  |
| color_wheel | orange | 8 | logistic | 0.6208 | 0.0485 | 78 | 1278 |  |  |
| color_wheel | orange | 14 | ridge | 0.9814 | 0.0861 | 78 | 1278 | <== | PASS |
| color_wheel | orange | 14 | lda | 0.9718 | 0.0823 | 78 | 1278 |  |  |
| color_wheel | orange | 14 | dom | 0.9172 | 0.0867 | 78 | 1278 |  |  |
| color_wheel | orange | 14 | logistic | 0.6111 | 0.0573 | 78 | 1278 |  |  |
| color_wheel | red | 6 | ridge | 0.9460 | 0.1173 | 149 | 1207 | <== | PASS |
| color_wheel | red | 6 | lda | 0.9116 | 0.1151 | 149 | 1207 |  |  |
| color_wheel | red | 6 | dom | 0.8902 | 0.1217 | 149 | 1207 |  |  |
| color_wheel | red | 6 | logistic | 0.6863 | 0.0832 | 149 | 1207 |  |  |
| color_wheel | red | 8 | ridge | 0.9504 | 0.1135 | 149 | 1207 | <== | PASS |
| color_wheel | red | 8 | lda | 0.9165 | 0.1102 | 149 | 1207 |  |  |
| color_wheel | red | 8 | dom | 0.8201 | 0.1185 | 149 | 1207 |  |  |
| color_wheel | red | 8 | logistic | 0.6386 | 0.0852 | 149 | 1207 |  |  |
| color_wheel | red | 14 | ridge | 0.9345 | 0.1153 | 149 | 1207 | <== | PASS |
| color_wheel | red | 14 | lda | 0.9026 | 0.1129 | 149 | 1207 |  |  |
| color_wheel | red | 14 | dom | 0.8398 | 0.1189 | 149 | 1207 |  |  |
| color_wheel | red | 14 | logistic | 0.6306 | 0.0827 | 149 | 1207 |  |  |
| color_wheel | red-orange | 6 | dom | 0.9870 | 0.0752 | 47 | 1309 | <== | PASS |
| color_wheel | red-orange | 6 | ridge | 0.9498 | 0.0735 | 47 | 1309 |  |  |
| color_wheel | red-orange | 6 | lda | 0.9477 | 0.0737 | 47 | 1309 |  |  |
| color_wheel | red-orange | 6 | logistic | 0.7436 | 0.0588 | 47 | 1309 |  |  |
| color_wheel | red-orange | 8 | dom | 0.9650 | 0.0743 | 47 | 1309 | <== | PASS |
| color_wheel | red-orange | 8 | lda | 0.9463 | 0.0707 | 47 | 1309 |  |  |
| color_wheel | red-orange | 8 | ridge | 0.9426 | 0.0705 | 47 | 1309 |  |  |
| color_wheel | red-orange | 8 | logistic | 0.7380 | 0.0614 | 47 | 1309 |  |  |
| color_wheel | red-orange | 14 | dom | 0.9784 | 0.0743 | 47 | 1309 | <== | PASS |
| color_wheel | red-orange | 14 | lda | 0.9352 | 0.0724 | 47 | 1309 |  |  |
| color_wheel | red-orange | 14 | ridge | 0.9347 | 0.0720 | 47 | 1309 |  |  |
| color_wheel | red-orange | 14 | logistic | 0.7263 | 0.0644 | 47 | 1309 |  |  |
| color_wheel | red-violet | 6 | dom | 0.8381 | 0.0363 | 7 | 1349 | <== |  |
| color_wheel | red-violet | 6 | lda | 0.6997 | 0.0329 | 7 | 1349 |  |  |
| color_wheel | red-violet | 6 | ridge | 0.6817 | 0.0325 | 7 | 1349 |  |  |
| color_wheel | red-violet | 6 | logistic | 0.4554 | 0.0092 | 7 | 1349 |  |  |
| color_wheel | red-violet | 8 | dom | 0.8235 | 0.0361 | 7 | 1349 | <== |  |
| color_wheel | red-violet | 8 | lda | 0.7403 | 0.0322 | 7 | 1349 |  |  |
| color_wheel | red-violet | 8 | ridge | 0.7354 | 0.0316 | 7 | 1349 |  |  |
| color_wheel | red-violet | 8 | logistic | 0.4503 | 0.0138 | 7 | 1349 |  |  |
| color_wheel | red-violet | 14 | dom | 0.8916 | 0.0363 | 7 | 1349 | <== |  |
| color_wheel | red-violet | 14 | lda | 0.6601 | 0.0322 | 7 | 1349 |  |  |
| color_wheel | red-violet | 14 | ridge | 0.6062 | 0.0314 | 7 | 1349 |  |  |
| color_wheel | red-violet | 14 | logistic | 0.4686 | 0.0116 | 7 | 1349 |  |  |
| color_wheel | violet | 6 | ridge | 0.9895 | 0.0361 | 14 | 1342 | <== | PASS |
| color_wheel | violet | 6 | lda | 0.9739 | 0.0358 | 14 | 1342 |  |  |
| color_wheel | violet | 6 | dom | 0.9248 | 0.0407 | 14 | 1342 |  |  |
| color_wheel | violet | 6 | logistic | 0.5827 | 0.0228 | 14 | 1342 |  |  |
| color_wheel | violet | 8 | ridge | 0.9918 | 0.0375 | 14 | 1342 | <== | PASS |
| color_wheel | violet | 8 | lda | 0.9771 | 0.0384 | 14 | 1342 |  |  |
| color_wheel | violet | 8 | dom | 0.8456 | 0.0408 | 14 | 1342 |  |  |
| color_wheel | violet | 8 | logistic | 0.5267 | 0.0235 | 14 | 1342 |  |  |
| color_wheel | violet | 14 | ridge | 0.9618 | 0.0397 | 14 | 1342 | <== | PASS |
| color_wheel | violet | 14 | lda | 0.9326 | 0.0402 | 14 | 1342 |  |  |
| color_wheel | violet | 14 | dom | 0.8343 | 0.0411 | 14 | 1342 |  |  |
| color_wheel | violet | 14 | logistic | 0.6489 | 0.0262 | 14 | 1342 |  |  |
| color_wheel | yellow | 6 | ridge | 0.9389 | 0.1052 | 157 | 1199 | <== | PASS |
| color_wheel | yellow | 6 | lda | 0.9169 | 0.1027 | 157 | 1199 |  |  |
| color_wheel | yellow | 6 | dom | 0.8513 | 0.1105 | 157 | 1199 |  |  |
| color_wheel | yellow | 6 | logistic | 0.6112 | 0.0717 | 157 | 1199 |  |  |
| color_wheel | yellow | 8 | ridge | 0.9471 | 0.1016 | 157 | 1199 | <== | PASS |
| color_wheel | yellow | 8 | lda | 0.9193 | 0.0997 | 157 | 1199 |  |  |
| color_wheel | yellow | 8 | dom | 0.8195 | 0.1103 | 157 | 1199 |  |  |
| color_wheel | yellow | 8 | logistic | 0.5675 | 0.0740 | 157 | 1199 |  |  |
| color_wheel | yellow | 14 | ridge | 0.9350 | 0.1044 | 157 | 1199 | <== | PASS |
| color_wheel | yellow | 14 | lda | 0.8836 | 0.1016 | 157 | 1199 |  |  |
| color_wheel | yellow | 14 | dom | 0.8507 | 0.1106 | 157 | 1199 |  |  |
| color_wheel | yellow | 14 | logistic | 0.6332 | 0.0684 | 157 | 1199 |  |  |
| color_wheel | yellow-green | 6 | dom | 0.9794 | 0.0743 | 44 | 1312 | <== | PASS |
| color_wheel | yellow-green | 6 | lda | 0.9621 | 0.0733 | 44 | 1312 |  |  |
| color_wheel | yellow-green | 6 | ridge | 0.9620 | 0.0733 | 44 | 1312 |  |  |
| color_wheel | yellow-green | 6 | logistic | 0.6884 | 0.0597 | 44 | 1312 |  |  |
| color_wheel | yellow-green | 8 | dom | 0.9458 | 0.0741 | 44 | 1312 | <== | PASS |
| color_wheel | yellow-green | 8 | ridge | 0.9338 | 0.0733 | 44 | 1312 |  |  |
| color_wheel | yellow-green | 8 | lda | 0.9281 | 0.0738 | 44 | 1312 |  |  |
| color_wheel | yellow-green | 8 | logistic | 0.7160 | 0.0624 | 44 | 1312 |  |  |
| color_wheel | yellow-green | 14 | dom | 0.9605 | 0.0735 | 44 | 1312 | <== | PASS |
| color_wheel | yellow-green | 14 | lda | 0.9381 | 0.0729 | 44 | 1312 |  |  |
| color_wheel | yellow-green | 14 | ridge | 0.9334 | 0.0724 | 44 | 1312 |  |  |
| color_wheel | yellow-green | 14 | logistic | 0.7215 | 0.0617 | 44 | 1312 |  |  |
| color_wheel | yellow-orange | 6 | ridge | 0.8247 | 0.0473 | 20 | 1336 | <== |  |
| color_wheel | yellow-orange | 6 | lda | 0.8185 | 0.0505 | 20 | 1336 |  |  |
| color_wheel | yellow-orange | 6 | dom | 0.7896 | 0.0567 | 20 | 1336 |  |  |
| color_wheel | yellow-orange | 6 | logistic | 0.6543 | 0.0181 | 20 | 1336 |  |  |
| color_wheel | yellow-orange | 8 | ridge | 0.8096 | 0.0545 | 20 | 1336 | <== |  |
| color_wheel | yellow-orange | 8 | dom | 0.8081 | 0.0556 | 20 | 1336 |  |  |
| color_wheel | yellow-orange | 8 | lda | 0.8034 | 0.0537 | 20 | 1336 |  |  |
| color_wheel | yellow-orange | 8 | logistic | 0.6188 | 0.0210 | 20 | 1336 |  |  |
| color_wheel | yellow-orange | 14 | dom | 0.8275 | 0.0555 | 20 | 1336 | <== |  |
| color_wheel | yellow-orange | 14 | ridge | 0.8097 | 0.0523 | 20 | 1336 |  |  |
| color_wheel | yellow-orange | 14 | lda | 0.7972 | 0.0520 | 20 | 1336 |  |  |
| color_wheel | yellow-orange | 14 | logistic | 0.7393 | 0.0276 | 20 | 1336 |  |  |
| continents | africa | 6 | dom | 0.9718 | 0.1117 | 76 | 907 | <== | PASS |
| continents | africa | 6 | ridge | 0.9590 | 0.1060 | 76 | 907 |  |  |
| continents | africa | 6 | lda | 0.9361 | 0.1022 | 76 | 907 |  |  |
| continents | africa | 6 | logistic | 0.6922 | 0.0740 | 76 | 907 |  |  |
| continents | africa | 8 | dom | 0.9614 | 0.1108 | 76 | 907 | <== | PASS |
| continents | africa | 8 | ridge | 0.9613 | 0.1125 | 76 | 907 |  |  |
| continents | africa | 8 | lda | 0.9008 | 0.1082 | 76 | 907 |  |  |
| continents | africa | 8 | logistic | 0.7090 | 0.0773 | 76 | 907 |  |  |
| continents | africa | 14 | dom | 0.9610 | 0.1120 | 76 | 907 | <== | PASS |
| continents | africa | 14 | ridge | 0.9481 | 0.1043 | 76 | 907 |  |  |
| continents | africa | 14 | lda | 0.9076 | 0.0988 | 76 | 907 |  |  |
| continents | africa | 14 | logistic | 0.6461 | 0.0693 | 76 | 907 |  |  |
| continents | asia | 6 | dom | 0.9545 | 0.1187 | 82 | 901 | <== | PASS |
| continents | asia | 6 | ridge | 0.9472 | 0.1139 | 82 | 901 |  |  |
| continents | asia | 6 | lda | 0.9186 | 0.1049 | 82 | 901 |  |  |
| continents | asia | 6 | logistic | 0.6520 | 0.0646 | 82 | 901 |  |  |
| continents | asia | 8 | ridge | 0.9543 | 0.1159 | 82 | 901 | <== | PASS |
| continents | asia | 8 | dom | 0.9468 | 0.1189 | 82 | 901 |  |  |
| continents | asia | 8 | lda | 0.9195 | 0.1068 | 82 | 901 |  |  |
| continents | asia | 8 | logistic | 0.6164 | 0.0616 | 82 | 901 |  |  |
| continents | asia | 14 | dom | 0.9587 | 0.1181 | 82 | 901 | <== | PASS |
| continents | asia | 14 | ridge | 0.9550 | 0.1123 | 82 | 901 |  |  |
| continents | asia | 14 | lda | 0.9244 | 0.0979 | 82 | 901 |  |  |
| continents | asia | 14 | logistic | 0.6476 | 0.0627 | 82 | 901 |  |  |
| continents | europe | 6 | ridge | 0.9586 | 0.1097 | 74 | 909 | <== | PASS |
| continents | europe | 6 | lda | 0.9502 | 0.1039 | 74 | 909 |  |  |
| continents | europe | 6 | dom | 0.9444 | 0.1101 | 74 | 909 |  |  |
| continents | europe | 6 | logistic | 0.5964 | 0.0683 | 74 | 909 |  |  |
| continents | europe | 8 | ridge | 0.9644 | 0.1131 | 74 | 909 | <== | PASS |
| continents | europe | 8 | dom | 0.9479 | 0.1140 | 74 | 909 |  |  |
| continents | europe | 8 | lda | 0.9432 | 0.1059 | 74 | 909 |  |  |
| continents | europe | 8 | logistic | 0.5944 | 0.0596 | 74 | 909 |  |  |
| continents | europe | 14 | ridge | 0.9551 | 0.1142 | 74 | 909 | <== | PASS |
| continents | europe | 14 | dom | 0.9511 | 0.1093 | 74 | 909 |  |  |
| continents | europe | 14 | lda | 0.9505 | 0.1052 | 74 | 909 |  |  |
| continents | europe | 14 | logistic | 0.6356 | 0.0674 | 74 | 909 |  |  |
| continents | north_america | 6 | ridge | 0.9596 | 0.1409 | 66 | 917 | <== | PASS |
| continents | north_america | 6 | lda | 0.9534 | 0.1264 | 66 | 917 |  |  |
| continents | north_america | 6 | dom | 0.9419 | 0.1467 | 66 | 917 |  |  |
| continents | north_america | 6 | logistic | 0.9316 | 0.0971 | 66 | 917 |  |  |
| continents | north_america | 8 | ridge | 0.9592 | 0.1437 | 66 | 917 | <== | PASS |
| continents | north_america | 8 | lda | 0.9532 | 0.1327 | 66 | 917 |  |  |
| continents | north_america | 8 | dom | 0.9430 | 0.1478 | 66 | 917 |  |  |
| continents | north_america | 8 | logistic | 0.6527 | 0.0729 | 66 | 917 |  |  |
| continents | north_america | 14 | ridge | 0.9597 | 0.1438 | 66 | 917 | <== | PASS |
| continents | north_america | 14 | lda | 0.9586 | 0.1306 | 66 | 917 |  |  |
| continents | north_america | 14 | dom | 0.9529 | 0.1490 | 66 | 917 |  |  |
| continents | north_america | 14 | logistic | 0.6679 | 0.0819 | 66 | 917 |  |  |
| continents | oceania | 6 | dom | 0.9777 | 0.0818 | 46 | 937 | <== | PASS |
| continents | oceania | 6 | ridge | 0.9688 | 0.0781 | 46 | 937 |  |  |
| continents | oceania | 6 | lda | 0.9655 | 0.0766 | 46 | 937 |  |  |
| continents | oceania | 6 | logistic | 0.9370 | 0.0735 | 46 | 937 |  |  |
| continents | oceania | 8 | dom | 0.9697 | 0.0843 | 46 | 937 | <== | PASS |
| continents | oceania | 8 | ridge | 0.9650 | 0.0786 | 46 | 937 |  |  |
| continents | oceania | 8 | lda | 0.9626 | 0.0770 | 46 | 937 |  |  |
| continents | oceania | 8 | logistic | 0.6486 | 0.0471 | 46 | 937 |  |  |
| continents | oceania | 14 | ridge | 0.9838 | 0.0787 | 46 | 937 | <== | PASS |
| continents | oceania | 14 | dom | 0.9785 | 0.0831 | 46 | 937 |  |  |
| continents | oceania | 14 | lda | 0.9780 | 0.0768 | 46 | 937 |  |  |
| continents | oceania | 14 | logistic | 0.6485 | 0.0501 | 46 | 937 |  |  |
| continents | south_america | 6 | dom | 0.9893 | 0.0935 | 55 | 928 | <== | PASS |
| continents | south_america | 6 | ridge | 0.9740 | 0.0862 | 55 | 928 |  |  |
| continents | south_america | 6 | lda | 0.9732 | 0.0866 | 55 | 928 |  |  |
| continents | south_america | 6 | logistic | 0.6947 | 0.0590 | 55 | 928 |  |  |
| continents | south_america | 8 | dom | 0.9879 | 0.0931 | 55 | 928 | <== | PASS |
| continents | south_america | 8 | ridge | 0.9793 | 0.0817 | 55 | 928 |  |  |
| continents | south_america | 8 | lda | 0.9777 | 0.0811 | 55 | 928 |  |  |
| continents | south_america | 8 | logistic | 0.6539 | 0.0528 | 55 | 928 |  |  |
| continents | south_america | 14 | dom | 0.9883 | 0.0927 | 55 | 928 | <== | PASS |
| continents | south_america | 14 | lda | 0.9740 | 0.0826 | 55 | 928 |  |  |
| continents | south_america | 14 | ridge | 0.9735 | 0.0813 | 55 | 928 |  |  |
| continents | south_america | 14 | logistic | 0.5877 | 0.0522 | 55 | 928 |  |  |
| costliness | costliness | 6 | dom | 0.8660 | 0.1199 | 25 | 706 | <== |  |
| costliness | costliness | 6 | ridge | 0.7222 | 0.1113 | 25 | 706 |  |  |
| costliness | costliness | 6 | lda | 0.7219 | 0.0972 | 25 | 706 |  |  |
| costliness | costliness | 6 | logistic | 0.5390 | 0.0564 | 25 | 706 |  |  |
| costliness | costliness | 8 | dom | 0.9131 | 0.1281 | 25 | 706 | <== | PASS |
| costliness | costliness | 8 | ridge | 0.7248 | 0.1056 | 25 | 706 |  |  |
| costliness | costliness | 8 | lda | 0.6374 | 0.0900 | 25 | 706 |  |  |
| costliness | costliness | 8 | logistic | 0.5273 | 0.0531 | 25 | 706 |  |  |
| costliness | costliness | 14 | dom | 0.8207 | 0.1235 | 25 | 706 | <== |  |
| costliness | costliness | 14 | ridge | 0.6951 | 0.1153 | 25 | 706 |  |  |
| costliness | costliness | 14 | lda | 0.6584 | 0.0943 | 25 | 706 |  |  |
| costliness | costliness | 14 | logistic | 0.6062 | 0.0610 | 25 | 706 |  |  |
| directions | east | 6 | lda | 0.9875 | 0.0693 | 36 | 1047 | <== | PASS |
| directions | east | 6 | ridge | 0.9867 | 0.0908 | 36 | 1047 |  |  |
| directions | east | 6 | dom | 0.9610 | 0.1054 | 36 | 1047 |  |  |
| directions | east | 6 | logistic | 0.5990 | 0.0332 | 36 | 1047 |  |  |
| directions | east | 8 | ridge | 0.9847 | 0.0888 | 36 | 1047 | <== | PASS |
| directions | east | 8 | lda | 0.9846 | 0.0704 | 36 | 1047 |  |  |
| directions | east | 8 | dom | 0.9243 | 0.1031 | 36 | 1047 |  |  |
| directions | east | 8 | logistic | 0.6228 | 0.0249 | 36 | 1047 |  |  |
| directions | east | 14 | lda | 0.9859 | 0.0717 | 36 | 1047 | <== | PASS |
| directions | east | 14 | ridge | 0.9852 | 0.0864 | 36 | 1047 |  |  |
| directions | east | 14 | dom | 0.9481 | 0.1023 | 36 | 1047 |  |  |
| directions | east | 14 | logistic | 0.6895 | 0.0248 | 36 | 1047 |  |  |
| directions | north | 6 | ridge | 0.9785 | 0.0967 | 36 | 1047 | <== | PASS |
| directions | north | 6 | lda | 0.9740 | 0.0701 | 36 | 1047 |  |  |
| directions | north | 6 | dom | 0.9663 | 0.1098 | 36 | 1047 |  |  |
| directions | north | 6 | logistic | 0.6676 | 0.0197 | 36 | 1047 |  |  |
| directions | north | 8 | ridge | 0.9809 | 0.0945 | 36 | 1047 | <== | PASS |
| directions | north | 8 | lda | 0.9783 | 0.0741 | 36 | 1047 |  |  |
| directions | north | 8 | dom | 0.9530 | 0.1105 | 36 | 1047 |  |  |
| directions | north | 8 | logistic | 0.6234 | 0.0304 | 36 | 1047 |  |  |
| directions | north | 14 | lda | 0.9799 | 0.0842 | 36 | 1047 | <== | PASS |
| directions | north | 14 | ridge | 0.9789 | 0.0994 | 36 | 1047 |  |  |
| directions | north | 14 | dom | 0.9690 | 0.1117 | 36 | 1047 |  |  |
| directions | north | 14 | logistic | 0.6471 | 0.0441 | 36 | 1047 |  |  |
| directions | northeast | 6 | dom | 0.9725 | 0.0707 | 33 | 1050 | <== | PASS |
| directions | northeast | 6 | lda | 0.9507 | 0.0573 | 33 | 1050 |  |  |
| directions | northeast | 6 | ridge | 0.9452 | 0.0569 | 33 | 1050 |  |  |
| directions | northeast | 6 | logistic | 0.7005 | 0.0501 | 33 | 1050 |  |  |
| directions | northeast | 8 | lda | 0.9632 | 0.0614 | 33 | 1050 | <== | PASS |
| directions | northeast | 8 | ridge | 0.9623 | 0.0602 | 33 | 1050 |  |  |
| directions | northeast | 8 | dom | 0.9543 | 0.0704 | 33 | 1050 |  |  |
| directions | northeast | 8 | logistic | 0.7010 | 0.0480 | 33 | 1050 |  |  |
| directions | northeast | 14 | dom | 0.9617 | 0.0708 | 33 | 1050 | <== | PASS |
| directions | northeast | 14 | lda | 0.9558 | 0.0620 | 33 | 1050 |  |  |
| directions | northeast | 14 | ridge | 0.9481 | 0.0614 | 33 | 1050 |  |  |
| directions | northeast | 14 | logistic | 0.5991 | 0.0450 | 33 | 1050 |  |  |
| directions | northwest | 6 | ridge | 0.9556 | 0.0525 | 31 | 1052 | <== | PASS |
| directions | northwest | 6 | lda | 0.9464 | 0.0471 | 31 | 1052 |  |  |
| directions | northwest | 6 | dom | 0.9459 | 0.0590 | 31 | 1052 |  |  |
| directions | northwest | 6 | logistic | 0.7097 | 0.0398 | 31 | 1052 |  |  |
| directions | northwest | 8 | ridge | 0.9494 | 0.0496 | 31 | 1052 | <== | PASS |
| directions | northwest | 8 | lda | 0.9324 | 0.0450 | 31 | 1052 |  |  |
| directions | northwest | 8 | dom | 0.9276 | 0.0576 | 31 | 1052 |  |  |
| directions | northwest | 8 | logistic | 0.6849 | 0.0386 | 31 | 1052 |  |  |
| directions | northwest | 14 | ridge | 0.9393 | 0.0510 | 31 | 1052 | <== | PASS |
| directions | northwest | 14 | dom | 0.9256 | 0.0583 | 31 | 1052 |  |  |
| directions | northwest | 14 | lda | 0.9220 | 0.0419 | 31 | 1052 |  |  |
| directions | northwest | 14 | logistic | 0.6421 | 0.0392 | 31 | 1052 |  |  |
| directions | south | 6 | ridge | 0.9675 | 0.0886 | 30 | 1053 | <== | PASS |
| directions | south | 6 | dom | 0.9502 | 0.1069 | 30 | 1053 |  |  |
| directions | south | 6 | lda | 0.9500 | 0.0588 | 30 | 1053 |  |  |
| directions | south | 6 | logistic | 0.6502 | 0.0351 | 30 | 1053 |  |  |
| directions | south | 8 | ridge | 0.9742 | 0.0908 | 30 | 1053 | <== | PASS |
| directions | south | 8 | lda | 0.9521 | 0.0688 | 30 | 1053 |  |  |
| directions | south | 8 | dom | 0.9339 | 0.1059 | 30 | 1053 |  |  |
| directions | south | 8 | logistic | 0.6771 | 0.0389 | 30 | 1053 |  |  |
| directions | south | 14 | ridge | 0.9731 | 0.0904 | 30 | 1053 | <== | PASS |
| directions | south | 14 | lda | 0.9582 | 0.0719 | 30 | 1053 |  |  |
| directions | south | 14 | dom | 0.9423 | 0.1066 | 30 | 1053 |  |  |
| directions | south | 14 | logistic | 0.7589 | 0.0415 | 30 | 1053 |  |  |
| directions | southeast | 6 | ridge | 0.9877 | 0.0571 | 23 | 1060 | <== | PASS |
| directions | southeast | 6 | lda | 0.9852 | 0.0524 | 23 | 1060 |  |  |
| directions | southeast | 6 | dom | 0.9714 | 0.0651 | 23 | 1060 |  |  |
| directions | southeast | 6 | logistic | 0.6251 | 0.0329 | 23 | 1060 |  |  |
| directions | southeast | 8 | ridge | 0.9839 | 0.0533 | 23 | 1060 | <== | PASS |
| directions | southeast | 8 | lda | 0.9784 | 0.0467 | 23 | 1060 |  |  |
| directions | southeast | 8 | dom | 0.9483 | 0.0632 | 23 | 1060 |  |  |
| directions | southeast | 8 | logistic | 0.5872 | 0.0287 | 23 | 1060 |  |  |
| directions | southeast | 14 | ridge | 0.9767 | 0.0561 | 23 | 1060 | <== | PASS |
| directions | southeast | 14 | lda | 0.9706 | 0.0513 | 23 | 1060 |  |  |
| directions | southeast | 14 | dom | 0.9527 | 0.0640 | 23 | 1060 |  |  |
| directions | southeast | 14 | logistic | 0.7029 | 0.0427 | 23 | 1060 |  |  |
| directions | southwest | 6 | ridge | 0.9837 | 0.0611 | 25 | 1058 | <== | PASS |
| directions | southwest | 6 | lda | 0.9807 | 0.0610 | 25 | 1058 |  |  |
| directions | southwest | 6 | dom | 0.9717 | 0.0673 | 25 | 1058 |  |  |
| directions | southwest | 6 | logistic | 0.6988 | 0.0481 | 25 | 1058 |  |  |
| directions | southwest | 8 | ridge | 0.9836 | 0.0593 | 25 | 1058 | <== | PASS |
| directions | southwest | 8 | lda | 0.9811 | 0.0582 | 25 | 1058 |  |  |
| directions | southwest | 8 | dom | 0.9626 | 0.0661 | 25 | 1058 |  |  |
| directions | southwest | 8 | logistic | 0.8092 | 0.0458 | 25 | 1058 |  |  |
| directions | southwest | 14 | ridge | 0.9853 | 0.0583 | 25 | 1058 | <== | PASS |
| directions | southwest | 14 | lda | 0.9829 | 0.0586 | 25 | 1058 |  |  |
| directions | southwest | 14 | dom | 0.9651 | 0.0665 | 25 | 1058 |  |  |
| directions | southwest | 14 | logistic | 0.7196 | 0.0535 | 25 | 1058 |  |  |
| directions | west | 6 | ridge | 0.9752 | 0.0763 | 25 | 1058 | <== | PASS |
| directions | west | 6 | lda | 0.9657 | 0.0546 | 25 | 1058 |  |  |
| directions | west | 6 | dom | 0.9526 | 0.0939 | 25 | 1058 |  |  |
| directions | west | 6 | logistic | 0.6379 | 0.0222 | 25 | 1058 |  |  |
| directions | west | 8 | ridge | 0.9768 | 0.0766 | 25 | 1058 | <== | PASS |
| directions | west | 8 | lda | 0.9762 | 0.0576 | 25 | 1058 |  |  |
| directions | west | 8 | dom | 0.9250 | 0.0912 | 25 | 1058 |  |  |
| directions | west | 8 | logistic | 0.5803 | 0.0297 | 25 | 1058 |  |  |
| directions | west | 14 | ridge | 0.9662 | 0.0698 | 25 | 1058 | <== | PASS |
| directions | west | 14 | lda | 0.9636 | 0.0520 | 25 | 1058 |  |  |
| directions | west | 14 | dom | 0.9385 | 0.0925 | 25 | 1058 |  |  |
| directions | west | 14 | logistic | 0.5407 | 0.0207 | 25 | 1058 |  |  |
| duration | duration | 6 | dom | 0.8985 | 0.2083 | 113 | 616 | <== |  |
| duration | duration | 6 | ridge | 0.8173 | 0.2050 | 113 | 616 |  |  |
| duration | duration | 6 | lda | 0.8064 | 0.1783 | 113 | 616 |  |  |
| duration | duration | 6 | logistic | 0.6970 | 0.1383 | 113 | 616 |  |  |
| duration | duration | 8 | dom | 0.9133 | 0.2146 | 113 | 616 | <== | PASS |
| duration | duration | 8 | ridge | 0.8406 | 0.1930 | 113 | 616 |  |  |
| duration | duration | 8 | lda | 0.8332 | 0.1716 | 113 | 616 |  |  |
| duration | duration | 8 | logistic | 0.6230 | 0.1331 | 113 | 616 |  |  |
| duration | duration | 14 | dom | 0.8803 | 0.2172 | 113 | 616 | <== |  |
| duration | duration | 14 | ridge | 0.8742 | 0.2013 | 113 | 616 |  |  |
| duration | duration | 14 | lda | 0.8460 | 0.1832 | 113 | 616 |  |  |
| duration | duration | 14 | logistic | 0.6125 | 0.1212 | 113 | 616 |  |  |
| harmfulness | harmfulness | 6 | dom | 0.8442 | 0.2306 | 105 | 619 | <== |  |
| harmfulness | harmfulness | 6 | ridge | 0.7716 | 0.2013 | 105 | 619 |  |  |
| harmfulness | harmfulness | 6 | lda | 0.7504 | 0.1666 | 105 | 619 |  |  |
| harmfulness | harmfulness | 6 | logistic | 0.7180 | 0.1240 | 105 | 619 |  |  |
| harmfulness | harmfulness | 8 | dom | 0.8781 | 0.2586 | 105 | 619 | <== |  |
| harmfulness | harmfulness | 8 | ridge | 0.7753 | 0.2056 | 105 | 619 |  |  |
| harmfulness | harmfulness | 8 | lda | 0.7577 | 0.1653 | 105 | 619 |  |  |
| harmfulness | harmfulness | 8 | logistic | 0.6669 | 0.0989 | 105 | 619 |  |  |
| harmfulness | harmfulness | 14 | dom | 0.8860 | 0.2442 | 105 | 619 | <== |  |
| harmfulness | harmfulness | 14 | ridge | 0.8240 | 0.2249 | 105 | 619 |  |  |
| harmfulness | harmfulness | 14 | lda | 0.8005 | 0.1770 | 105 | 619 |  |  |
| harmfulness | harmfulness | 14 | logistic | 0.7283 | 0.1389 | 105 | 619 |  |  |
| location_type | indoors | 6 | dom | 0.9074 | 0.1496 | 58 | 662 | <== | PASS |
| location_type | indoors | 6 | ridge | 0.8592 | 0.1303 | 58 | 662 |  |  |
| location_type | indoors | 6 | lda | 0.8575 | 0.1163 | 58 | 662 |  |  |
| location_type | indoors | 6 | logistic | 0.5979 | 0.0623 | 58 | 662 |  |  |
| location_type | indoors | 8 | dom | 0.9360 | 0.1552 | 58 | 662 | <== | PASS |
| location_type | indoors | 8 | ridge | 0.8887 | 0.1257 | 58 | 662 |  |  |
| location_type | indoors | 8 | lda | 0.8777 | 0.1191 | 58 | 662 |  |  |
| location_type | indoors | 8 | logistic | 0.5734 | 0.0632 | 58 | 662 |  |  |
| location_type | indoors | 14 | dom | 0.8931 | 0.1446 | 58 | 662 | <== |  |
| location_type | indoors | 14 | ridge | 0.8619 | 0.1287 | 58 | 662 |  |  |
| location_type | indoors | 14 | lda | 0.8598 | 0.1174 | 58 | 662 |  |  |
| location_type | indoors | 14 | logistic | 0.5422 | 0.0740 | 58 | 662 |  |  |
| location_type | outdoors | 6 | dom | 0.8912 | 0.1969 | 115 | 605 | <== |  |
| location_type | outdoors | 6 | ridge | 0.8240 | 0.1668 | 115 | 605 |  |  |
| location_type | outdoors | 6 | lda | 0.8078 | 0.1513 | 115 | 605 |  |  |
| location_type | outdoors | 6 | logistic | 0.6824 | 0.1030 | 115 | 605 |  |  |
| location_type | outdoors | 8 | dom | 0.9119 | 0.2115 | 115 | 605 | <== | PASS |
| location_type | outdoors | 8 | ridge | 0.8251 | 0.1669 | 115 | 605 |  |  |
| location_type | outdoors | 8 | lda | 0.8228 | 0.1543 | 115 | 605 |  |  |
| location_type | outdoors | 8 | logistic | 0.6212 | 0.0933 | 115 | 605 |  |  |
| location_type | outdoors | 14 | dom | 0.9010 | 0.2056 | 115 | 605 | <== | PASS |
| location_type | outdoors | 14 | lda | 0.8434 | 0.1604 | 115 | 605 |  |  |
| location_type | outdoors | 14 | ridge | 0.8396 | 0.1739 | 115 | 605 |  |  |
| location_type | outdoors | 14 | logistic | 0.6493 | 0.0960 | 115 | 605 |  |  |
| lovingness | lovingness | 6 | dom | 0.9044 | 0.1416 | 59 | 698 | <== | PASS |
| lovingness | lovingness | 6 | lda | 0.8219 | 0.1379 | 59 | 698 |  |  |
| lovingness | lovingness | 6 | ridge | 0.8066 | 0.1598 | 59 | 698 |  |  |
| lovingness | lovingness | 6 | logistic | 0.6947 | 0.0963 | 59 | 698 |  |  |
| lovingness | lovingness | 8 | dom | 0.9190 | 0.1542 | 59 | 698 | <== | PASS |
| lovingness | lovingness | 8 | lda | 0.8303 | 0.1308 | 59 | 698 |  |  |
| lovingness | lovingness | 8 | ridge | 0.8285 | 0.1487 | 59 | 698 |  |  |
| lovingness | lovingness | 8 | logistic | 0.6460 | 0.0676 | 59 | 698 |  |  |
| lovingness | lovingness | 14 | dom | 0.8985 | 0.1485 | 59 | 698 | <== |  |
| lovingness | lovingness | 14 | ridge | 0.8573 | 0.1776 | 59 | 698 |  |  |
| lovingness | lovingness | 14 | lda | 0.8445 | 0.1577 | 59 | 698 |  |  |
| lovingness | lovingness | 14 | logistic | 0.6295 | 0.0945 | 59 | 698 |  |  |
| months | april | 6 | lda | 0.9857 | 0.0878 | 75 | 1290 | <== | PASS |
| months | april | 6 | ridge | 0.9787 | 0.0876 | 75 | 1290 |  |  |
| months | april | 6 | dom | 0.9167 | 0.0749 | 75 | 1290 |  |  |
| months | april | 6 | logistic | 0.7443 | 0.0785 | 75 | 1290 |  |  |
| months | april | 8 | lda | 0.9912 | 0.0876 | 75 | 1290 | <== | PASS |
| months | april | 8 | ridge | 0.9849 | 0.0870 | 75 | 1290 |  |  |
| months | april | 8 | dom | 0.8936 | 0.0771 | 75 | 1290 |  |  |
| months | april | 8 | logistic | 0.6725 | 0.0720 | 75 | 1290 |  |  |
| months | april | 14 | lda | 0.9767 | 0.0878 | 75 | 1290 | <== | PASS |
| months | april | 14 | ridge | 0.9756 | 0.0869 | 75 | 1290 |  |  |
| months | april | 14 | dom | 0.9049 | 0.0783 | 75 | 1290 |  |  |
| months | april | 14 | logistic | 0.7521 | 0.0760 | 75 | 1290 |  |  |
| months | august | 6 | lda | 0.9869 | 0.0722 | 59 | 1306 | <== | PASS |
| months | august | 6 | ridge | 0.9868 | 0.0697 | 59 | 1306 |  |  |
| months | august | 6 | dom | 0.9219 | 0.0684 | 59 | 1306 |  |  |
| months | august | 6 | logistic | 0.7445 | 0.0708 | 59 | 1306 |  |  |
| months | august | 8 | ridge | 0.9876 | 0.0693 | 59 | 1306 | <== | PASS |
| months | august | 8 | lda | 0.9850 | 0.0719 | 59 | 1306 |  |  |
| months | august | 8 | dom | 0.9142 | 0.0699 | 59 | 1306 |  |  |
| months | august | 8 | logistic | 0.7682 | 0.0632 | 59 | 1306 |  |  |
| months | august | 14 | lda | 0.9866 | 0.0716 | 59 | 1306 | <== | PASS |
| months | august | 14 | ridge | 0.9821 | 0.0674 | 59 | 1306 |  |  |
| months | august | 14 | dom | 0.9073 | 0.0701 | 59 | 1306 |  |  |
| months | august | 14 | logistic | 0.7885 | 0.0665 | 59 | 1306 |  |  |
| months | december | 6 | ridge | 0.9880 | 0.0839 | 66 | 1299 | <== | PASS |
| months | december | 6 | lda | 0.9870 | 0.0862 | 66 | 1299 |  |  |
| months | december | 6 | dom | 0.9583 | 0.0742 | 66 | 1299 |  |  |
| months | december | 6 | logistic | 0.7086 | 0.0717 | 66 | 1299 |  |  |
| months | december | 8 | ridge | 0.9902 | 0.0817 | 66 | 1299 | <== | PASS |
| months | december | 8 | lda | 0.9860 | 0.0833 | 66 | 1299 |  |  |
| months | december | 8 | logistic | 0.9494 | 0.0841 | 66 | 1299 |  |  |
| months | december | 8 | dom | 0.9114 | 0.0766 | 66 | 1299 |  |  |
| months | december | 14 | ridge | 0.9872 | 0.0793 | 66 | 1299 | <== | PASS |
| months | december | 14 | lda | 0.9856 | 0.0832 | 66 | 1299 |  |  |
| months | december | 14 | dom | 0.9521 | 0.0789 | 66 | 1299 |  |  |
| months | december | 14 | logistic | 0.6861 | 0.0728 | 66 | 1299 |  |  |
| months | february | 6 | lda | 0.9857 | 0.0840 | 76 | 1289 | <== | PASS |
| months | february | 6 | ridge | 0.9816 | 0.0885 | 76 | 1289 |  |  |
| months | february | 6 | dom | 0.9324 | 0.0770 | 76 | 1289 |  |  |
| months | february | 6 | logistic | 0.6059 | 0.0680 | 76 | 1289 |  |  |
| months | february | 8 | lda | 0.9853 | 0.0840 | 76 | 1289 | <== | PASS |
| months | february | 8 | ridge | 0.9827 | 0.0845 | 76 | 1289 |  |  |
| months | february | 8 | dom | 0.9059 | 0.0786 | 76 | 1289 |  |  |
| months | february | 8 | logistic | 0.6492 | 0.0705 | 76 | 1289 |  |  |
| months | february | 14 | lda | 0.9814 | 0.0805 | 76 | 1289 | <== | PASS |
| months | february | 14 | ridge | 0.9736 | 0.0816 | 76 | 1289 |  |  |
| months | february | 14 | dom | 0.9445 | 0.0818 | 76 | 1289 |  |  |
| months | february | 14 | logistic | 0.6228 | 0.0651 | 76 | 1289 |  |  |
| months | january | 6 | ridge | 0.9892 | 0.0699 | 70 | 1295 | <== | PASS |
| months | january | 6 | lda | 0.9886 | 0.0710 | 70 | 1295 |  |  |
| months | january | 6 | dom | 0.9488 | 0.0687 | 70 | 1295 |  |  |
| months | january | 6 | logistic | 0.8118 | 0.0612 | 70 | 1295 |  |  |
| months | january | 8 | ridge | 0.9905 | 0.0676 | 70 | 1295 | <== | PASS |
| months | january | 8 | lda | 0.9885 | 0.0682 | 70 | 1295 |  |  |
| months | january | 8 | dom | 0.9107 | 0.0674 | 70 | 1295 |  |  |
| months | january | 8 | logistic | 0.7221 | 0.0486 | 70 | 1295 |  |  |
| months | january | 14 | lda | 0.9881 | 0.0699 | 70 | 1295 | <== | PASS |
| months | january | 14 | ridge | 0.9876 | 0.0689 | 70 | 1295 |  |  |
| months | january | 14 | dom | 0.9385 | 0.0685 | 70 | 1295 |  |  |
| months | january | 14 | logistic | 0.6906 | 0.0589 | 70 | 1295 |  |  |
| months | july | 6 | ridge | 0.9922 | 0.0628 | 64 | 1301 | <== | PASS |
| months | july | 6 | lda | 0.9880 | 0.0614 | 64 | 1301 |  |  |
| months | july | 6 | dom | 0.9282 | 0.0677 | 64 | 1301 |  |  |
| months | july | 6 | logistic | 0.7625 | 0.0493 | 64 | 1301 |  |  |
| months | july | 8 | ridge | 0.9853 | 0.0637 | 64 | 1301 | <== | PASS |
| months | july | 8 | lda | 0.9806 | 0.0636 | 64 | 1301 |  |  |
| months | july | 8 | dom | 0.8866 | 0.0693 | 64 | 1301 |  |  |
| months | july | 8 | logistic | 0.6711 | 0.0481 | 64 | 1301 |  |  |
| months | july | 14 | ridge | 0.9854 | 0.0679 | 64 | 1301 | <== | PASS |
| months | july | 14 | lda | 0.9853 | 0.0689 | 64 | 1301 |  |  |
| months | july | 14 | dom | 0.9289 | 0.0702 | 64 | 1301 |  |  |
| months | july | 14 | logistic | 0.6847 | 0.0494 | 64 | 1301 |  |  |
| months | june | 6 | ridge | 0.9727 | 0.0765 | 86 | 1279 | <== | PASS |
| months | june | 6 | lda | 0.9500 | 0.0745 | 86 | 1279 |  |  |
| months | june | 6 | dom | 0.9125 | 0.0782 | 86 | 1279 |  |  |
| months | june | 6 | logistic | 0.7113 | 0.0619 | 86 | 1279 |  |  |
| months | june | 8 | ridge | 0.9834 | 0.0783 | 86 | 1279 | <== | PASS |
| months | june | 8 | lda | 0.9726 | 0.0784 | 86 | 1279 |  |  |
| months | june | 8 | dom | 0.9117 | 0.0794 | 86 | 1279 |  |  |
| months | june | 8 | logistic | 0.6707 | 0.0487 | 86 | 1279 |  |  |
| months | june | 14 | ridge | 0.9716 | 0.0800 | 86 | 1279 | <== | PASS |
| months | june | 14 | lda | 0.9532 | 0.0794 | 86 | 1279 |  |  |
| months | june | 14 | dom | 0.9061 | 0.0808 | 86 | 1279 |  |  |
| months | june | 14 | logistic | 0.7169 | 0.0495 | 86 | 1279 |  |  |
| months | march | 6 | ridge | 0.9294 | 0.0956 | 75 | 1290 | <== | PASS |
| months | march | 6 | lda | 0.9274 | 0.0951 | 75 | 1290 |  |  |
| months | march | 6 | dom | 0.9116 | 0.0817 | 75 | 1290 |  |  |
| months | march | 6 | logistic | 0.6923 | 0.0833 | 75 | 1290 |  |  |
| months | march | 8 | lda | 0.9586 | 0.0941 | 75 | 1290 | <== | PASS |
| months | march | 8 | ridge | 0.9550 | 0.0942 | 75 | 1290 |  |  |
| months | march | 8 | dom | 0.9039 | 0.0836 | 75 | 1290 |  |  |
| months | march | 8 | logistic | 0.7204 | 0.0764 | 75 | 1290 |  |  |
| months | march | 14 | lda | 0.9207 | 0.0955 | 75 | 1290 | <== | PASS |
| months | march | 14 | ridge | 0.9026 | 0.0957 | 75 | 1290 |  |  |
| months | march | 14 | dom | 0.8960 | 0.0859 | 75 | 1290 |  |  |
| months | march | 14 | logistic | 0.7453 | 0.0744 | 75 | 1290 |  |  |
| months | may | 6 | ridge | 0.9646 | 0.0865 | 81 | 1284 | <== | PASS |
| months | may | 6 | lda | 0.9532 | 0.0825 | 81 | 1284 |  |  |
| months | may | 6 | dom | 0.9188 | 0.0778 | 81 | 1284 |  |  |
| months | may | 6 | logistic | 0.7337 | 0.0793 | 81 | 1284 |  |  |
| months | may | 8 | ridge | 0.9660 | 0.0894 | 81 | 1284 | <== | PASS |
| months | may | 8 | lda | 0.9611 | 0.0868 | 81 | 1284 |  |  |
| months | may | 8 | dom | 0.9009 | 0.0804 | 81 | 1284 |  |  |
| months | may | 8 | logistic | 0.7810 | 0.0710 | 81 | 1284 |  |  |
| months | may | 14 | ridge | 0.9462 | 0.0897 | 81 | 1284 | <== | PASS |
| months | may | 14 | lda | 0.9374 | 0.0879 | 81 | 1284 |  |  |
| months | may | 14 | dom | 0.8936 | 0.0865 | 81 | 1284 |  |  |
| months | may | 14 | logistic | 0.6591 | 0.0708 | 81 | 1284 |  |  |
| months | november | 6 | lda | 0.9869 | 0.0840 | 56 | 1309 | <== | PASS |
| months | november | 6 | ridge | 0.9857 | 0.0831 | 56 | 1309 |  |  |
| months | november | 6 | dom | 0.9193 | 0.0737 | 56 | 1309 |  |  |
| months | november | 6 | logistic | 0.6481 | 0.0710 | 56 | 1309 |  |  |
| months | november | 8 | lda | 0.9846 | 0.0829 | 56 | 1309 | <== | PASS |
| months | november | 8 | ridge | 0.9842 | 0.0854 | 56 | 1309 |  |  |
| months | november | 8 | dom | 0.9077 | 0.0748 | 56 | 1309 |  |  |
| months | november | 8 | logistic | 0.6343 | 0.0640 | 56 | 1309 |  |  |
| months | november | 14 | lda | 0.9796 | 0.0837 | 56 | 1309 | <== | PASS |
| months | november | 14 | ridge | 0.9773 | 0.0829 | 56 | 1309 |  |  |
| months | november | 14 | dom | 0.9287 | 0.0752 | 56 | 1309 |  |  |
| months | november | 14 | logistic | 0.6744 | 0.0660 | 56 | 1309 |  |  |
| months | october | 6 | lda | 0.9898 | 0.0826 | 65 | 1300 | <== | PASS |
| months | october | 6 | ridge | 0.9851 | 0.0815 | 65 | 1300 |  |  |
| months | october | 6 | dom | 0.9224 | 0.0726 | 65 | 1300 |  |  |
| months | october | 6 | logistic | 0.6869 | 0.0676 | 65 | 1300 |  |  |
| months | october | 8 | lda | 0.9898 | 0.0799 | 65 | 1300 | <== | PASS |
| months | october | 8 | ridge | 0.9887 | 0.0778 | 65 | 1300 |  |  |
| months | october | 8 | dom | 0.9157 | 0.0725 | 65 | 1300 |  |  |
| months | october | 8 | logistic | 0.6996 | 0.0620 | 65 | 1300 |  |  |
| months | october | 14 | lda | 0.9809 | 0.0826 | 65 | 1300 | <== | PASS |
| months | october | 14 | ridge | 0.9729 | 0.0825 | 65 | 1300 |  |  |
| months | october | 14 | dom | 0.9293 | 0.0727 | 65 | 1300 |  |  |
| months | october | 14 | logistic | 0.6925 | 0.0638 | 65 | 1300 |  |  |
| months | september | 6 | lda | 0.9842 | 0.0728 | 60 | 1305 | <== | PASS |
| months | september | 6 | ridge | 0.9792 | 0.0682 | 60 | 1305 |  |  |
| months | september | 6 | dom | 0.8942 | 0.0707 | 60 | 1305 |  |  |
| months | september | 6 | logistic | 0.7167 | 0.0484 | 60 | 1305 |  |  |
| months | september | 8 | lda | 0.9806 | 0.0742 | 60 | 1305 | <== | PASS |
| months | september | 8 | ridge | 0.9766 | 0.0716 | 60 | 1305 |  |  |
| months | september | 8 | dom | 0.8885 | 0.0729 | 60 | 1305 |  |  |
| months | september | 8 | logistic | 0.6860 | 0.0535 | 60 | 1305 |  |  |
| months | september | 14 | lda | 0.9774 | 0.0776 | 60 | 1305 | <== | PASS |
| months | september | 14 | ridge | 0.9738 | 0.0741 | 60 | 1305 |  |  |
| months | september | 14 | dom | 0.8983 | 0.0722 | 60 | 1305 |  |  |
| months | september | 14 | logistic | 0.7096 | 0.0626 | 60 | 1305 |  |  |
| moon_phases | first_quarter | 6 | dom | 0.9986 | 0.0989 | 25 | 867 | <== | PASS |
| moon_phases | first_quarter | 6 | ridge | 0.9957 | 0.0909 | 25 | 867 |  |  |
| moon_phases | first_quarter | 6 | lda | 0.9922 | 0.0619 | 25 | 867 |  |  |
| moon_phases | first_quarter | 6 | logistic | 0.8720 | 0.0226 | 25 | 867 |  |  |
| moon_phases | first_quarter | 8 | ridge | 0.9952 | 0.0937 | 25 | 867 | <== | PASS |
| moon_phases | first_quarter | 8 | lda | 0.9943 | 0.0733 | 25 | 867 |  |  |
| moon_phases | first_quarter | 8 | dom | 0.9792 | 0.0861 | 25 | 867 |  |  |
| moon_phases | first_quarter | 8 | logistic | 0.8616 | 0.0367 | 25 | 867 |  |  |
| moon_phases | first_quarter | 14 | lda | 0.9987 | 0.0718 | 25 | 867 | <== | PASS |
| moon_phases | first_quarter | 14 | ridge | 0.9978 | 0.0861 | 25 | 867 |  |  |
| moon_phases | first_quarter | 14 | dom | 0.9880 | 0.0961 | 25 | 867 |  |  |
| moon_phases | first_quarter | 14 | logistic | 0.8090 | 0.0458 | 25 | 867 |  |  |
| moon_phases | full_moon | 6 | dom | 0.9877 | 0.1159 | 90 | 802 | <== | PASS |
| moon_phases | full_moon | 6 | lda | 0.9864 | 0.1059 | 90 | 802 |  |  |
| moon_phases | full_moon | 6 | ridge | 0.9851 | 0.1048 | 90 | 802 |  |  |
| moon_phases | full_moon | 6 | logistic | 0.7225 | 0.0756 | 90 | 802 |  |  |
| moon_phases | full_moon | 8 | lda | 0.9907 | 0.1056 | 90 | 802 | <== | PASS |
| moon_phases | full_moon | 8 | ridge | 0.9894 | 0.1038 | 90 | 802 |  |  |
| moon_phases | full_moon | 8 | dom | 0.9649 | 0.1161 | 90 | 802 |  |  |
| moon_phases | full_moon | 8 | logistic | 0.6308 | 0.0687 | 90 | 802 |  |  |
| moon_phases | full_moon | 14 | lda | 0.9863 | 0.1071 | 90 | 802 | <== | PASS |
| moon_phases | full_moon | 14 | ridge | 0.9857 | 0.1068 | 90 | 802 |  |  |
| moon_phases | full_moon | 14 | dom | 0.9767 | 0.1167 | 90 | 802 |  |  |
| moon_phases | full_moon | 14 | logistic | 0.6873 | 0.0827 | 90 | 802 |  |  |
| moon_phases | last_quarter | 6 | ridge | 0.9947 | 0.0771 | 25 | 867 | <== | PASS |
| moon_phases | last_quarter | 6 | lda | 0.9942 | 0.0649 | 25 | 867 |  |  |
| moon_phases | last_quarter | 6 | dom | 0.9900 | 0.0901 | 25 | 867 |  |  |
| moon_phases | last_quarter | 6 | logistic | 0.8284 | 0.0514 | 25 | 867 |  |  |
| moon_phases | last_quarter | 8 | lda | 0.9987 | 0.0683 | 25 | 867 | <== | PASS |
| moon_phases | last_quarter | 8 | ridge | 0.9975 | 0.0774 | 25 | 867 |  |  |
| moon_phases | last_quarter | 8 | dom | 0.9758 | 0.0791 | 25 | 867 |  |  |
| moon_phases | last_quarter | 8 | logistic | 0.8406 | 0.0515 | 25 | 867 |  |  |
| moon_phases | last_quarter | 14 | ridge | 0.9986 | 0.0776 | 25 | 867 | <== | PASS |
| moon_phases | last_quarter | 14 | lda | 0.9986 | 0.0648 | 25 | 867 |  |  |
| moon_phases | last_quarter | 14 | dom | 0.9773 | 0.0856 | 25 | 867 |  |  |
| moon_phases | last_quarter | 14 | logistic | 0.7167 | 0.0527 | 25 | 867 |  |  |
| moon_phases | new_moon | 6 | dom | 0.9929 | 0.1105 | 73 | 819 | <== | PASS |
| moon_phases | new_moon | 6 | lda | 0.9928 | 0.1084 | 73 | 819 |  |  |
| moon_phases | new_moon | 6 | ridge | 0.9923 | 0.1089 | 73 | 819 |  |  |
| moon_phases | new_moon | 6 | logistic | 0.7892 | 0.1006 | 73 | 819 |  |  |
| moon_phases | new_moon | 8 | lda | 0.9954 | 0.1105 | 73 | 819 | <== | PASS |
| moon_phases | new_moon | 8 | ridge | 0.9945 | 0.1090 | 73 | 819 |  |  |
| moon_phases | new_moon | 8 | dom | 0.9747 | 0.1170 | 73 | 819 |  |  |
| moon_phases | new_moon | 8 | logistic | 0.7980 | 0.0962 | 73 | 819 |  |  |
| moon_phases | new_moon | 14 | ridge | 0.9940 | 0.1145 | 73 | 819 | <== | PASS |
| moon_phases | new_moon | 14 | lda | 0.9937 | 0.1158 | 73 | 819 |  |  |
| moon_phases | new_moon | 14 | dom | 0.9834 | 0.1193 | 73 | 819 |  |  |
| moon_phases | new_moon | 14 | logistic | 0.8317 | 0.0942 | 73 | 819 |  |  |
| moon_phases | waning_crescent | 6 | lda | 0.9905 | 0.0502 | 15 | 877 | <== | PASS |
| moon_phases | waning_crescent | 6 | dom | 0.9878 | 0.0506 | 15 | 877 |  |  |
| moon_phases | waning_crescent | 6 | ridge | 0.9868 | 0.0503 | 15 | 877 |  |  |
| moon_phases | waning_crescent | 6 | logistic | 0.7945 | 0.0421 | 15 | 877 |  |  |
| moon_phases | waning_crescent | 8 | ridge | 0.9936 | 0.0489 | 15 | 877 | <== | PASS |
| moon_phases | waning_crescent | 8 | lda | 0.9933 | 0.0499 | 15 | 877 |  |  |
| moon_phases | waning_crescent | 8 | dom | 0.9699 | 0.0489 | 15 | 877 |  |  |
| moon_phases | waning_crescent | 8 | logistic | 0.7216 | 0.0367 | 15 | 877 |  |  |
| moon_phases | waning_crescent | 14 | lda | 0.9871 | 0.0489 | 15 | 877 | <== | PASS |
| moon_phases | waning_crescent | 14 | ridge | 0.9845 | 0.0454 | 15 | 877 |  |  |
| moon_phases | waning_crescent | 14 | dom | 0.9743 | 0.0504 | 15 | 877 |  |  |
| moon_phases | waning_crescent | 14 | logistic | 0.8484 | 0.0457 | 15 | 877 |  |  |
| moon_phases | waning_gibbous | 6 | dom | 0.9967 | 0.0654 | 16 | 876 | <== | PASS |
| moon_phases | waning_gibbous | 6 | lda | 0.9922 | 0.0657 | 16 | 876 |  |  |
| moon_phases | waning_gibbous | 6 | ridge | 0.9904 | 0.0651 | 16 | 876 |  |  |
| moon_phases | waning_gibbous | 6 | logistic | 0.7699 | 0.0553 | 16 | 876 |  |  |
| moon_phases | waning_gibbous | 8 | ridge | 0.9931 | 0.0628 | 16 | 876 | <== | PASS |
| moon_phases | waning_gibbous | 8 | lda | 0.9929 | 0.0632 | 16 | 876 |  |  |
| moon_phases | waning_gibbous | 8 | dom | 0.9610 | 0.0644 | 16 | 876 |  |  |
| moon_phases | waning_gibbous | 8 | logistic | 0.8355 | 0.0549 | 16 | 876 |  |  |
| moon_phases | waning_gibbous | 14 | lda | 0.9935 | 0.0650 | 16 | 876 | <== | PASS |
| moon_phases | waning_gibbous | 14 | ridge | 0.9923 | 0.0626 | 16 | 876 |  |  |
| moon_phases | waning_gibbous | 14 | dom | 0.9709 | 0.0661 | 16 | 876 |  |  |
| moon_phases | waning_gibbous | 14 | logistic | 0.7168 | 0.0412 | 16 | 876 |  |  |
| moon_phases | waxing_crescent | 6 | dom | 0.9941 | 0.0674 | 23 | 869 | <== | PASS |
| moon_phases | waxing_crescent | 6 | ridge | 0.9927 | 0.0633 | 23 | 869 |  |  |
| moon_phases | waxing_crescent | 6 | lda | 0.9923 | 0.0648 | 23 | 869 |  |  |
| moon_phases | waxing_crescent | 6 | logistic | 0.7796 | 0.0601 | 23 | 869 |  |  |
| moon_phases | waxing_crescent | 8 | lda | 0.9928 | 0.0658 | 23 | 869 | <== | PASS |
| moon_phases | waxing_crescent | 8 | ridge | 0.9923 | 0.0634 | 23 | 869 |  |  |
| moon_phases | waxing_crescent | 8 | dom | 0.9921 | 0.0674 | 23 | 869 |  |  |
| moon_phases | waxing_crescent | 8 | logistic | 0.6771 | 0.0554 | 23 | 869 |  |  |
| moon_phases | waxing_crescent | 14 | lda | 0.9937 | 0.0663 | 23 | 869 | <== | PASS |
| moon_phases | waxing_crescent | 14 | ridge | 0.9932 | 0.0648 | 23 | 869 |  |  |
| moon_phases | waxing_crescent | 14 | dom | 0.9900 | 0.0682 | 23 | 869 |  |  |
| moon_phases | waxing_crescent | 14 | logistic | 0.7139 | 0.0554 | 23 | 869 |  |  |
| moon_phases | waxing_gibbous | 6 | dom | 0.9880 | 0.0678 | 17 | 875 | <== | PASS |
| moon_phases | waxing_gibbous | 6 | ridge | 0.9825 | 0.0656 | 17 | 875 |  |  |
| moon_phases | waxing_gibbous | 6 | lda | 0.9810 | 0.0650 | 17 | 875 |  |  |
| moon_phases | waxing_gibbous | 6 | logistic | 0.8877 | 0.0524 | 17 | 875 |  |  |
| moon_phases | waxing_gibbous | 8 | lda | 0.9739 | 0.0634 | 17 | 875 | <== | PASS |
| moon_phases | waxing_gibbous | 8 | ridge | 0.9672 | 0.0622 | 17 | 875 |  |  |
| moon_phases | waxing_gibbous | 8 | dom | 0.9636 | 0.0666 | 17 | 875 |  |  |
| moon_phases | waxing_gibbous | 8 | logistic | 0.8166 | 0.0516 | 17 | 875 |  |  |
| moon_phases | waxing_gibbous | 14 | lda | 0.9880 | 0.0626 | 17 | 875 | <== | PASS |
| moon_phases | waxing_gibbous | 14 | ridge | 0.9851 | 0.0610 | 17 | 875 |  |  |
| moon_phases | waxing_gibbous | 14 | dom | 0.9759 | 0.0686 | 17 | 875 |  |  |
| moon_phases | waxing_gibbous | 14 | logistic | 0.6639 | 0.0539 | 17 | 875 |  |  |
| physical_size | physical_size | 6 | dom | 0.8845 | 0.1611 | 33 | 679 | <== |  |
| physical_size | physical_size | 6 | lda | 0.8134 | 0.1151 | 33 | 679 |  |  |
| physical_size | physical_size | 6 | ridge | 0.8060 | 0.1264 | 33 | 679 |  |  |
| physical_size | physical_size | 6 | logistic | 0.6789 | 0.0962 | 33 | 679 |  |  |
| physical_size | physical_size | 8 | dom | 0.9304 | 0.1743 | 33 | 679 | <== | PASS |
| physical_size | physical_size | 8 | lda | 0.7908 | 0.1244 | 33 | 679 |  |  |
| physical_size | physical_size | 8 | ridge | 0.7838 | 0.1390 | 33 | 679 |  |  |
| physical_size | physical_size | 8 | logistic | 0.6359 | 0.0821 | 33 | 679 |  |  |
| physical_size | physical_size | 14 | dom | 0.8991 | 0.1634 | 33 | 679 | <== |  |
| physical_size | physical_size | 14 | lda | 0.8135 | 0.1132 | 33 | 679 |  |  |
| physical_size | physical_size | 14 | ridge | 0.7549 | 0.1240 | 33 | 679 |  |  |
| physical_size | physical_size | 14 | logistic | 0.6437 | 0.0790 | 33 | 679 |  |  |
| seasons | autumn | 6 | lda | 0.9840 | 0.0693 | 50 | 785 | <== | PASS |
| seasons | autumn | 6 | dom | 0.9793 | 0.0857 | 50 | 785 |  |  |
| seasons | autumn | 6 | ridge | 0.9756 | 0.0801 | 50 | 785 |  |  |
| seasons | autumn | 6 | logistic | 0.7191 | 0.0439 | 50 | 785 |  |  |
| seasons | autumn | 8 | lda | 0.9828 | 0.0683 | 50 | 785 | <== | PASS |
| seasons | autumn | 8 | ridge | 0.9777 | 0.0749 | 50 | 785 |  |  |
| seasons | autumn | 8 | dom | 0.9769 | 0.0828 | 50 | 785 |  |  |
| seasons | autumn | 8 | logistic | 0.7019 | 0.0411 | 50 | 785 |  |  |
| seasons | autumn | 14 | dom | 0.9660 | 0.0831 | 50 | 785 | <== | PASS |
| seasons | autumn | 14 | ridge | 0.9183 | 0.0731 | 50 | 785 |  |  |
| seasons | autumn | 14 | lda | 0.9107 | 0.0712 | 50 | 785 |  |  |
| seasons | autumn | 14 | logistic | 0.6581 | 0.0462 | 50 | 785 |  |  |
| seasons | spring | 6 | ridge | 0.9908 | 0.0867 | 67 | 768 | <== | PASS |
| seasons | spring | 6 | lda | 0.9906 | 0.0803 | 67 | 768 |  |  |
| seasons | spring | 6 | dom | 0.9900 | 0.0959 | 67 | 768 |  |  |
| seasons | spring | 6 | logistic | 0.7587 | 0.0595 | 67 | 768 |  |  |
| seasons | spring | 8 | ridge | 0.9942 | 0.0837 | 67 | 768 | <== | PASS |
| seasons | spring | 8 | lda | 0.9942 | 0.0773 | 67 | 768 |  |  |
| seasons | spring | 8 | dom | 0.9919 | 0.0933 | 67 | 768 |  |  |
| seasons | spring | 8 | logistic | 0.7069 | 0.0555 | 67 | 768 |  |  |
| seasons | spring | 14 | lda | 0.9899 | 0.0824 | 67 | 768 | <== | PASS |
| seasons | spring | 14 | ridge | 0.9886 | 0.0891 | 67 | 768 |  |  |
| seasons | spring | 14 | dom | 0.9855 | 0.0956 | 67 | 768 |  |  |
| seasons | spring | 14 | logistic | 0.6489 | 0.0576 | 67 | 768 |  |  |
| seasons | summer | 6 | lda | 0.9870 | 0.1054 | 97 | 738 | <== | PASS |
| seasons | summer | 6 | ridge | 0.9869 | 0.1050 | 97 | 738 |  |  |
| seasons | summer | 6 | dom | 0.9816 | 0.1108 | 97 | 738 |  |  |
| seasons | summer | 6 | logistic | 0.7328 | 0.0901 | 97 | 738 |  |  |
| seasons | summer | 8 | lda | 0.9913 | 0.1048 | 97 | 738 | <== | PASS |
| seasons | summer | 8 | ridge | 0.9909 | 0.1047 | 97 | 738 |  |  |
| seasons | summer | 8 | dom | 0.9780 | 0.1164 | 97 | 738 |  |  |
| seasons | summer | 8 | logistic | 0.6418 | 0.0802 | 97 | 738 |  |  |
| seasons | summer | 14 | lda | 0.9874 | 0.1096 | 97 | 738 | <== | PASS |
| seasons | summer | 14 | ridge | 0.9856 | 0.1106 | 97 | 738 |  |  |
| seasons | summer | 14 | dom | 0.9776 | 0.1183 | 97 | 738 |  |  |
| seasons | summer | 14 | logistic | 0.7221 | 0.0870 | 97 | 738 |  |  |
| seasons | winter | 6 | ridge | 0.9850 | 0.1154 | 89 | 746 | <== | PASS |
| seasons | winter | 6 | lda | 0.9847 | 0.1154 | 89 | 746 |  |  |
| seasons | winter | 6 | dom | 0.9787 | 0.1172 | 89 | 746 |  |  |
| seasons | winter | 6 | logistic | 0.6879 | 0.0883 | 89 | 746 |  |  |
| seasons | winter | 8 | lda | 0.9853 | 0.1136 | 89 | 746 | <== | PASS |
| seasons | winter | 8 | ridge | 0.9848 | 0.1157 | 89 | 746 |  |  |
| seasons | winter | 8 | dom | 0.9799 | 0.1219 | 89 | 746 |  |  |
| seasons | winter | 8 | logistic | 0.7136 | 0.0867 | 89 | 746 |  |  |
| seasons | winter | 14 | lda | 0.9808 | 0.1181 | 89 | 746 | <== | PASS |
| seasons | winter | 14 | ridge | 0.9804 | 0.1176 | 89 | 746 |  |  |
| seasons | winter | 14 | dom | 0.9739 | 0.1215 | 89 | 746 |  |  |
| seasons | winter | 14 | logistic | 0.7092 | 0.0853 | 89 | 746 |  |  |
| weekdays | friday | 6 | ridge | 0.9710 | 0.0661 | 61 | 913 | <== | PASS |
| weekdays | friday | 6 | lda | 0.9642 | 0.0659 | 61 | 913 |  |  |
| weekdays | friday | 6 | dom | 0.8980 | 0.0666 | 61 | 913 |  |  |
| weekdays | friday | 6 | logistic | 0.6237 | 0.0513 | 61 | 913 |  |  |
| weekdays | friday | 8 | ridge | 0.9666 | 0.0641 | 61 | 913 | <== | PASS |
| weekdays | friday | 8 | lda | 0.9661 | 0.0646 | 61 | 913 |  |  |
| weekdays | friday | 8 | dom | 0.9269 | 0.0677 | 61 | 913 |  |  |
| weekdays | friday | 8 | logistic | 0.6500 | 0.0549 | 61 | 913 |  |  |
| weekdays | friday | 14 | lda | 0.9723 | 0.0622 | 61 | 913 | <== | PASS |
| weekdays | friday | 14 | ridge | 0.9693 | 0.0620 | 61 | 913 |  |  |
| weekdays | friday | 14 | dom | 0.9263 | 0.0671 | 61 | 913 |  |  |
| weekdays | friday | 14 | logistic | 0.6315 | 0.0537 | 61 | 913 |  |  |
| weekdays | monday | 6 | lda | 0.9781 | 0.0597 | 50 | 924 | <== | PASS |
| weekdays | monday | 6 | ridge | 0.9760 | 0.0604 | 50 | 924 |  |  |
| weekdays | monday | 6 | dom | 0.9522 | 0.0619 | 50 | 924 |  |  |
| weekdays | monday | 6 | logistic | 0.6941 | 0.0540 | 50 | 924 |  |  |
| weekdays | monday | 8 | ridge | 0.9875 | 0.0617 | 50 | 924 | <== | PASS |
| weekdays | monday | 8 | lda | 0.9836 | 0.0618 | 50 | 924 |  |  |
| weekdays | monday | 8 | dom | 0.9588 | 0.0627 | 50 | 924 |  |  |
| weekdays | monday | 8 | logistic | 0.6386 | 0.0527 | 50 | 924 |  |  |
| weekdays | monday | 14 | ridge | 0.9873 | 0.0606 | 50 | 924 | <== | PASS |
| weekdays | monday | 14 | lda | 0.9814 | 0.0603 | 50 | 924 |  |  |
| weekdays | monday | 14 | dom | 0.9569 | 0.0620 | 50 | 924 |  |  |
| weekdays | monday | 14 | logistic | 0.7624 | 0.0561 | 50 | 924 |  |  |
| weekdays | saturday | 6 | ridge | 0.9943 | 0.0622 | 55 | 919 | <== | PASS |
| weekdays | saturday | 6 | lda | 0.9940 | 0.0612 | 55 | 919 |  |  |
| weekdays | saturday | 6 | dom | 0.9837 | 0.0623 | 55 | 919 |  |  |
| weekdays | saturday | 6 | logistic | 0.8310 | 0.0483 | 55 | 919 |  |  |
| weekdays | saturday | 8 | ridge | 0.9936 | 0.0611 | 55 | 919 | <== | PASS |
| weekdays | saturday | 8 | lda | 0.9933 | 0.0604 | 55 | 919 |  |  |
| weekdays | saturday | 8 | dom | 0.9813 | 0.0621 | 55 | 919 |  |  |
| weekdays | saturday | 8 | logistic | 0.7434 | 0.0490 | 55 | 919 |  |  |
| weekdays | saturday | 14 | ridge | 0.9949 | 0.0625 | 55 | 919 | <== | PASS |
| weekdays | saturday | 14 | lda | 0.9947 | 0.0619 | 55 | 919 |  |  |
| weekdays | saturday | 14 | dom | 0.9811 | 0.0617 | 55 | 919 |  |  |
| weekdays | saturday | 14 | logistic | 0.7863 | 0.0539 | 55 | 919 |  |  |
| weekdays | sunday | 6 | ridge | 0.9958 | 0.0430 | 20 | 954 | <== | PASS |
| weekdays | sunday | 6 | lda | 0.9954 | 0.0406 | 20 | 954 |  |  |
| weekdays | sunday | 6 | dom | 0.9872 | 0.0464 | 20 | 954 |  |  |
| weekdays | sunday | 6 | logistic | 0.7731 | 0.0272 | 20 | 954 |  |  |
| weekdays | sunday | 8 | ridge | 0.9959 | 0.0459 | 20 | 954 | <== | PASS |
| weekdays | sunday | 8 | lda | 0.9954 | 0.0430 | 20 | 954 |  |  |
| weekdays | sunday | 8 | dom | 0.9627 | 0.0464 | 20 | 954 |  |  |
| weekdays | sunday | 8 | logistic | 0.7692 | 0.0347 | 20 | 954 |  |  |
| weekdays | sunday | 14 | lda | 0.9958 | 0.0431 | 20 | 954 | <== | PASS |
| weekdays | sunday | 14 | ridge | 0.9952 | 0.0446 | 20 | 954 |  |  |
| weekdays | sunday | 14 | dom | 0.9765 | 0.0464 | 20 | 954 |  |  |
| weekdays | sunday | 14 | logistic | 0.7152 | 0.0341 | 20 | 954 |  |  |
| weekdays | thursday | 6 | ridge | 0.9761 | 0.0578 | 40 | 934 | <== | PASS |
| weekdays | thursday | 6 | lda | 0.9649 | 0.0581 | 40 | 934 |  |  |
| weekdays | thursday | 6 | dom | 0.8920 | 0.0595 | 40 | 934 |  |  |
| weekdays | thursday | 6 | logistic | 0.6976 | 0.0495 | 40 | 934 |  |  |
| weekdays | thursday | 8 | ridge | 0.9687 | 0.0581 | 40 | 934 | <== | PASS |
| weekdays | thursday | 8 | lda | 0.9502 | 0.0584 | 40 | 934 |  |  |
| weekdays | thursday | 8 | dom | 0.8660 | 0.0601 | 40 | 934 |  |  |
| weekdays | thursday | 8 | logistic | 0.7079 | 0.0488 | 40 | 934 |  |  |
| weekdays | thursday | 14 | ridge | 0.9735 | 0.0571 | 40 | 934 | <== | PASS |
| weekdays | thursday | 14 | lda | 0.9558 | 0.0565 | 40 | 934 |  |  |
| weekdays | thursday | 14 | dom | 0.8889 | 0.0590 | 40 | 934 |  |  |
| weekdays | thursday | 14 | logistic | 0.7310 | 0.0489 | 40 | 934 |  |  |
| weekdays | tuesday | 6 | ridge | 0.9776 | 0.0642 | 50 | 924 | <== | PASS |
| weekdays | tuesday | 6 | lda | 0.9677 | 0.0646 | 50 | 924 |  |  |
| weekdays | tuesday | 6 | dom | 0.9407 | 0.0676 | 50 | 924 |  |  |
| weekdays | tuesday | 6 | logistic | 0.7827 | 0.0599 | 50 | 924 |  |  |
| weekdays | tuesday | 8 | ridge | 0.9815 | 0.0671 | 50 | 924 | <== | PASS |
| weekdays | tuesday | 8 | lda | 0.9702 | 0.0672 | 50 | 924 |  |  |
| weekdays | tuesday | 8 | dom | 0.9259 | 0.0679 | 50 | 924 |  |  |
| weekdays | tuesday | 8 | logistic | 0.7087 | 0.0556 | 50 | 924 |  |  |
| weekdays | tuesday | 14 | ridge | 0.9877 | 0.0667 | 50 | 924 | <== | PASS |
| weekdays | tuesday | 14 | lda | 0.9838 | 0.0667 | 50 | 924 |  |  |
| weekdays | tuesday | 14 | dom | 0.9450 | 0.0677 | 50 | 924 |  |  |
| weekdays | tuesday | 14 | logistic | 0.7684 | 0.0612 | 50 | 924 |  |  |
| weekdays | wednesday | 6 | ridge | 0.9867 | 0.0608 | 51 | 923 | <== | PASS |
| weekdays | wednesday | 6 | lda | 0.9808 | 0.0610 | 51 | 923 |  |  |
| weekdays | wednesday | 6 | dom | 0.9367 | 0.0613 | 51 | 923 |  |  |
| weekdays | wednesday | 6 | logistic | 0.7296 | 0.0526 | 51 | 923 |  |  |
| weekdays | wednesday | 8 | ridge | 0.9840 | 0.0605 | 51 | 923 | <== | PASS |
| weekdays | wednesday | 8 | lda | 0.9766 | 0.0610 | 51 | 923 |  |  |
| weekdays | wednesday | 8 | dom | 0.9026 | 0.0612 | 51 | 923 |  |  |
| weekdays | wednesday | 8 | logistic | 0.7177 | 0.0525 | 51 | 923 |  |  |
| weekdays | wednesday | 14 | ridge | 0.9896 | 0.0605 | 51 | 923 | <== | PASS |
| weekdays | wednesday | 14 | lda | 0.9864 | 0.0609 | 51 | 923 |  |  |
| weekdays | wednesday | 14 | dom | 0.9380 | 0.0610 | 51 | 923 |  |  |
| weekdays | wednesday | 14 | logistic | 0.7244 | 0.0481 | 51 | 923 |  |  |

## Concept-level verdict (all-3-layers constraint)

| family | concept | L6 auroc | L8 auroc | L14 auroc | status |
|---|---|---|---|---|---|
| color_wheel | blue | 0.9498 | 0.9508 | 0.9376 | SURVIVOR |
| color_wheel | blue-green | 0.9833 | 0.9478 | 0.9543 | SURVIVOR |
| color_wheel | blue-violet | 0.9131 | 0.8895 | 0.9253 | NEAR-MISS/DROP (min_auroc=0.8895) |
| color_wheel | green | 0.9485 | 0.9649 | 0.9607 | SURVIVOR |
| color_wheel | orange | 0.9861 | 0.9908 | 0.9814 | SURVIVOR |
| color_wheel | red | 0.9460 | 0.9504 | 0.9345 | SURVIVOR |
| color_wheel | red-orange | 0.9870 | 0.9650 | 0.9784 | SURVIVOR |
| color_wheel | red-violet | 0.8381 | 0.8235 | 0.8916 | NEAR-MISS/DROP (min_auroc=0.8235) |
| color_wheel | violet | 0.9895 | 0.9918 | 0.9618 | SURVIVOR |
| color_wheel | yellow | 0.9389 | 0.9471 | 0.9350 | SURVIVOR |
| color_wheel | yellow-green | 0.9794 | 0.9458 | 0.9605 | SURVIVOR |
| color_wheel | yellow-orange | 0.8247 | 0.8096 | 0.8275 | NEAR-MISS/DROP (min_auroc=0.8096) |
| continents | africa | 0.9718 | 0.9614 | 0.9610 | SURVIVOR |
| continents | asia | 0.9545 | 0.9543 | 0.9587 | SURVIVOR |
| continents | europe | 0.9586 | 0.9644 | 0.9551 | SURVIVOR |
| continents | north_america | 0.9596 | 0.9592 | 0.9597 | SURVIVOR |
| continents | oceania | 0.9777 | 0.9697 | 0.9838 | SURVIVOR |
| continents | south_america | 0.9893 | 0.9879 | 0.9883 | SURVIVOR |
| costliness | costliness | 0.8660 | 0.9131 | 0.8207 | NEAR-MISS/DROP (min_auroc=0.8207) |
| directions | east | 0.9875 | 0.9847 | 0.9859 | SURVIVOR |
| directions | north | 0.9785 | 0.9809 | 0.9799 | SURVIVOR |
| directions | northeast | 0.9725 | 0.9632 | 0.9617 | SURVIVOR |
| directions | northwest | 0.9556 | 0.9494 | 0.9393 | SURVIVOR |
| directions | south | 0.9675 | 0.9742 | 0.9731 | SURVIVOR |
| directions | southeast | 0.9877 | 0.9839 | 0.9767 | SURVIVOR |
| directions | southwest | 0.9837 | 0.9836 | 0.9853 | SURVIVOR |
| directions | west | 0.9752 | 0.9768 | 0.9662 | SURVIVOR |
| duration | duration | 0.8985 | 0.9133 | 0.8803 | NEAR-MISS/DROP (min_auroc=0.8803) |
| harmfulness | harmfulness | 0.8442 | 0.8781 | 0.8860 | NEAR-MISS/DROP (min_auroc=0.8442) |
| location_type | indoors | 0.9074 | 0.9360 | 0.8931 | NEAR-MISS/DROP (min_auroc=0.8931) |
| location_type | outdoors | 0.8912 | 0.9119 | 0.9010 | NEAR-MISS/DROP (min_auroc=0.8912) |
| lovingness | lovingness | 0.9044 | 0.9190 | 0.8985 | NEAR-MISS/DROP (min_auroc=0.8985) |
| months | april | 0.9857 | 0.9912 | 0.9767 | SURVIVOR |
| months | august | 0.9869 | 0.9876 | 0.9866 | SURVIVOR |
| months | december | 0.9880 | 0.9902 | 0.9872 | SURVIVOR |
| months | february | 0.9857 | 0.9853 | 0.9814 | SURVIVOR |
| months | january | 0.9892 | 0.9905 | 0.9881 | SURVIVOR |
| months | july | 0.9922 | 0.9853 | 0.9854 | SURVIVOR |
| months | june | 0.9727 | 0.9834 | 0.9716 | SURVIVOR |
| months | march | 0.9294 | 0.9586 | 0.9207 | SURVIVOR |
| months | may | 0.9646 | 0.9660 | 0.9462 | SURVIVOR |
| months | november | 0.9869 | 0.9846 | 0.9796 | SURVIVOR |
| months | october | 0.9898 | 0.9898 | 0.9809 | SURVIVOR |
| months | september | 0.9842 | 0.9806 | 0.9774 | SURVIVOR |
| moon_phases | first_quarter | 0.9986 | 0.9952 | 0.9987 | SURVIVOR |
| moon_phases | full_moon | 0.9877 | 0.9907 | 0.9863 | SURVIVOR |
| moon_phases | last_quarter | 0.9947 | 0.9987 | 0.9986 | SURVIVOR |
| moon_phases | new_moon | 0.9929 | 0.9954 | 0.9940 | SURVIVOR |
| moon_phases | waning_crescent | 0.9905 | 0.9936 | 0.9871 | SURVIVOR |
| moon_phases | waning_gibbous | 0.9967 | 0.9931 | 0.9935 | SURVIVOR |
| moon_phases | waxing_crescent | 0.9941 | 0.9928 | 0.9937 | SURVIVOR |
| moon_phases | waxing_gibbous | 0.9880 | 0.9739 | 0.9880 | SURVIVOR |
| physical_size | physical_size | 0.8845 | 0.9304 | 0.8991 | NEAR-MISS/DROP (min_auroc=0.8845) |
| seasons | autumn | 0.9840 | 0.9828 | 0.9660 | SURVIVOR |
| seasons | spring | 0.9908 | 0.9942 | 0.9899 | SURVIVOR |
| seasons | summer | 0.9870 | 0.9913 | 0.9874 | SURVIVOR |
| seasons | winter | 0.9850 | 0.9853 | 0.9808 | SURVIVOR |
| weekdays | friday | 0.9710 | 0.9666 | 0.9723 | SURVIVOR |
| weekdays | monday | 0.9781 | 0.9875 | 0.9873 | SURVIVOR |
| weekdays | saturday | 0.9943 | 0.9936 | 0.9949 | SURVIVOR |
| weekdays | sunday | 0.9958 | 0.9959 | 0.9958 | SURVIVOR |
| weekdays | thursday | 0.9761 | 0.9687 | 0.9735 | SURVIVOR |
| weekdays | tuesday | 0.9776 | 0.9815 | 0.9877 | SURVIVOR |
| weekdays | wednesday | 0.9867 | 0.9840 | 0.9896 | SURVIVOR |

## Axis-convention verification (vs stage6_1 dose_calib.json, independent)

- color_wheel/blue@L14: t mine=0.129422 ref=0.129422 (rel err 0.00e+00); s95 mine=1.649950 ref=1.649950 (rel err 0.00e+00)
- color_wheel/blue-green@L14: t mine=0.166948 ref=0.166948 (rel err 0.00e+00); s95 mine=1.554839 ref=1.554839 (rel err 0.00e+00)
