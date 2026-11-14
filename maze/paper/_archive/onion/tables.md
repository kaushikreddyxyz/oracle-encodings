12 runs: ['edge-dummy fixed', 'edge-dummy-cont1 fixed', 'onion-checker trainable', 'onion-checker fixed', 'onion-checker-cont1 fixed', 'onion-checker-cont1 trainable', 'onion-checker-dummy fixed', 'onion-checker-dummy-cont1 fixed', 'onion-full trainable', 'onion-full fixed', 'onion-full-cont1 trainable', 'onion-full-cont1 fixed']

| model | world | p_sp | p_valid | p_invalid | solved | reached |
|---|---|---|---|---|---|---|
| edge-dummy fixed | true | 0.993 | 0.006 | 0.001 | 0.956 | 0.992 |
| edge-dummy fixed | phantom | 0.948 | 0.048 | 0.005 | 0.718 | 0.948 |
| edge-dummy-cont1 fixed | true | 0.998 | 0.001 | 0.000 | 0.990 | 1.000 |
| edge-dummy-cont1 fixed | phantom | 0.968 | 0.029 | 0.003 | 0.853 | 0.987 |
| onion-checker trainable | true | 0.973 | 0.026 | 0.001 | 0.803 | 0.980 |
| onion-checker trainable | phantom | 0.788 | 0.164 | 0.048 | 0.071 | 0.939 |
| onion-checker fixed | true | 0.976 | 0.023 | 0.001 | 0.828 | 0.985 |
| onion-checker fixed | phantom | 0.794 | 0.160 | 0.045 | 0.092 | 0.934 |
| onion-checker-cont1 fixed | true | 0.997 | 0.002 | 0.000 | 0.987 | 1.000 |
| onion-checker-cont1 fixed | phantom | 0.775 | 0.175 | 0.050 | 0.061 | 0.972 |
| onion-checker-cont1 trainable | true | 0.997 | 0.002 | 0.000 | 0.983 | 0.999 |
| onion-checker-cont1 trainable | phantom | 0.755 | 0.192 | 0.053 | 0.026 | 0.972 |
| onion-checker-dummy fixed | true | 0.964 | 0.030 | 0.006 | 0.731 | 0.978 |
| onion-checker-dummy fixed | phantom | 0.868 | 0.098 | 0.034 | 0.305 | 0.928 |
| onion-checker-dummy-cont1 fixed | true | 0.996 | 0.003 | 0.001 | 0.971 | 0.999 |
| onion-checker-dummy-cont1 fixed | phantom | 0.916 | 0.063 | 0.021 | 0.542 | 0.970 |
| onion-full trainable | true | 0.972 | 0.026 | 0.002 | 0.825 | 0.979 |
| onion-full trainable | phantom | 0.791 | 0.146 | 0.063 | 0.086 | 0.905 |
| onion-full fixed | true | 0.968 | 0.029 | 0.003 | 0.785 | 0.982 |
| onion-full fixed | phantom | 0.792 | 0.141 | 0.067 | 0.096 | 0.901 |
| onion-full-cont1 trainable | true | 0.995 | 0.004 | 0.001 | 0.976 | 0.995 |
| onion-full-cont1 trainable | phantom | 0.789 | 0.164 | 0.047 | 0.092 | 0.963 |
| onion-full-cont1 fixed | true | 0.995 | 0.004 | 0.000 | 0.983 | 0.998 |
| onion-full-cont1 fixed | phantom | 0.816 | 0.146 | 0.038 | 0.146 | 0.954 |
| REF base (mean of 5 seeds) | true | 0.971 | 0.028 | 0.001 | 0.795 | 0.982 |
| REF fixed (mean of 5 seeds) | phantom | 0.908 | 0.085 | 0.007 | 0.480 | 0.949 |
| REF fixed (mean of 5 seeds) | true | 0.975 | 0.024 | 0.001 | 0.838 | 0.983 |
| REF trainable (mean of 5 seeds) | phantom | 0.913 | 0.080 | 0.007 | 0.519 | 0.945 |
| REF trainable (mean of 5 seeds) | true | 0.974 | 0.025 | 0.001 | 0.828 | 0.981 | 

| model | n_models | same_maze | route_changed | route_changed_follows_prompt |
|---|---|---|---|---|
| onion-checker trainable | 1 | 0.800 | 0.065 | 0.274 |
| onion-checker fixed | 1 | 0.839 | 0.083 | 0.240 |
| onion-checker-cont1 fixed | 1 | 0.986 | 0.041 | 0.521 |
| onion-checker-cont1 trainable | 1 | 0.983 | 0.027 | 0.657 |
| onion-full trainable | 1 | 0.822 | 0.178 | 0.070 |
| onion-full fixed | 1 | 0.782 | 0.195 | 0.058 |
| onion-full-cont1 trainable | 1 | 0.975 | 0.092 | 0.284 |
| onion-full-cont1 fixed | 1 | 0.983 | 0.145 | 0.189 |
| REF edge fixed | 5 | 0.838 | 0.716 | 0.009 |
| REF edge trainable | 5 | 0.832 | 0.747 | 0.006 | 

| model | stratified_by | depth | n_items | p_new | top1_new | gain_nats | drop_nats | used_door | crossed_wall | solved_new |
|---|---|---|---|---|---|---|---|---|---|---|
| onion-checker trainable | door e+ | 0 | 1254 | 0.387 | 0.380 | 4.528 | 1.429 | 0.387 | 0.597 | 0.317 |
| onion-checker trainable | door e+ | 1 | 1723 | 0.220 | 0.214 | 2.283 | 0.673 | 0.132 | 0.726 | 0.117 |
| onion-checker trainable | door e+ | 2 | 1493 | 0.231 | 0.224 | 2.013 | 1.161 | 0.153 | 0.673 | 0.133 |
| onion-checker trainable | door e+ | 3 | 995 | 0.223 | 0.209 | 0.916 | 0.790 | 0.083 | 0.704 | 0.069 |
| onion-checker trainable | door e+ | all | 5465 | 0.262 | 0.254 | 2.475 | 1.001 | 0.187 | 0.678 | 0.159 |
| onion-checker trainable | wall e- | 0 | 1193 | 0.450 | 0.452 | 4.425 | 3.117 | 0.359 | 0.213 | 0.313 |
| onion-checker trainable | wall e- | 1 | 1706 | 0.225 | 0.218 | 2.350 | 0.589 | 0.147 | 0.767 | 0.128 |
| onion-checker trainable | wall e- | 2 | 1579 | 0.209 | 0.199 | 2.006 | 0.354 | 0.155 | 0.824 | 0.123 |
| onion-checker trainable | wall e- | 3 | 987 | 0.184 | 0.164 | 1.087 | 0.191 | 0.102 | 0.853 | 0.081 |
| onion-checker trainable | wall e- | all | 5465 | 0.262 | 0.254 | 2.475 | 1.001 | 0.187 | 0.678 | 0.159 |
| REF edge trainable (same labels) | door e+ | 0 | 1254 | 0.813 | 0.832 | 6.688 | 5.961 | 0.848 | 0.059 | 0.799 |
| REF edge trainable (same labels) | door e+ | 1 | 1723 | 0.831 | 0.852 | 7.007 | 6.196 | 0.867 | 0.054 | 0.825 |
| REF edge trainable (same labels) | door e+ | 2 | 1493 | 0.822 | 0.840 | 7.094 | 6.140 | 0.861 | 0.056 | 0.819 |
| REF edge trainable (same labels) | door e+ | 3 | 995 | 0.789 | 0.807 | 5.979 | 5.566 | 0.834 | 0.078 | 0.778 |
| REF edge trainable (same labels) | door e+ | all | 5465 | 0.817 | 0.836 | 6.771 | 6.012 | 0.855 | 0.060 | 0.809 |
| REF edge trainable (same labels) | wall e- | 0 | 1193 | 0.813 | 0.834 | 6.600 | 5.904 | 0.860 | 0.064 | 0.807 |
| REF edge trainable (same labels) | wall e- | 1 | 1706 | 0.829 | 0.845 | 7.135 | 6.266 | 0.859 | 0.059 | 0.820 |
| REF edge trainable (same labels) | wall e- | 2 | 1579 | 0.826 | 0.846 | 6.775 | 6.103 | 0.864 | 0.054 | 0.816 |
| REF edge trainable (same labels) | wall e- | 3 | 987 | 0.785 | 0.806 | 6.339 | 5.557 | 0.828 | 0.067 | 0.781 |
| REF edge trainable (same labels) | wall e- | all | 5465 | 0.817 | 0.836 | 6.771 | 6.012 | 0.855 | 0.060 | 0.809 |
| onion-checker fixed | door e+ | 0 | 1254 | 0.405 | 0.381 | 4.524 | 1.630 | 0.385 | 0.570 | 0.331 |
| onion-checker fixed | door e+ | 1 | 1723 | 0.302 | 0.304 | 3.019 | 1.253 | 0.250 | 0.636 | 0.223 |
| onion-checker fixed | door e+ | 2 | 1493 | 0.229 | 0.218 | 2.091 | 1.101 | 0.146 | 0.676 | 0.129 |
| onion-checker fixed | door e+ | 3 | 995 | 0.222 | 0.206 | 1.078 | 0.721 | 0.080 | 0.694 | 0.070 |
| onion-checker fixed | door e+ | all | 5465 | 0.291 | 0.280 | 2.757 | 1.201 | 0.222 | 0.642 | 0.195 |
| onion-checker fixed | wall e- | 0 | 1193 | 0.428 | 0.436 | 4.190 | 2.652 | 0.340 | 0.275 | 0.303 |
| onion-checker fixed | wall e- | 1 | 1706 | 0.318 | 0.307 | 3.245 | 1.375 | 0.248 | 0.618 | 0.226 |
| onion-checker fixed | wall e- | 2 | 1579 | 0.222 | 0.205 | 2.056 | 0.508 | 0.170 | 0.813 | 0.143 |
| onion-checker fixed | wall e- | 3 | 987 | 0.189 | 0.167 | 1.303 | 0.256 | 0.116 | 0.855 | 0.091 |
| onion-checker fixed | wall e- | all | 5465 | 0.291 | 0.280 | 2.757 | 1.201 | 0.222 | 0.642 | 0.195 |
| REF edge fixed (same labels) | door e+ | 0 | 1254 | 0.771 | 0.782 | 6.587 | 5.644 | 0.801 | 0.089 | 0.752 |
| REF edge fixed (same labels) | door e+ | 1 | 1723 | 0.802 | 0.813 | 7.089 | 5.812 | 0.828 | 0.084 | 0.786 |
| REF edge fixed (same labels) | door e+ | 2 | 1493 | 0.786 | 0.802 | 7.018 | 5.842 | 0.817 | 0.080 | 0.778 |
| REF edge fixed (same labels) | door e+ | 3 | 995 | 0.751 | 0.760 | 5.951 | 5.268 | 0.781 | 0.105 | 0.727 |
| REF edge fixed (same labels) | door e+ | all | 5465 | 0.781 | 0.793 | 6.747 | 5.683 | 0.810 | 0.088 | 0.765 |
| REF edge fixed (same labels) | wall e- | 0 | 1193 | 0.772 | 0.784 | 6.580 | 5.469 | 0.805 | 0.100 | 0.758 |
| REF edge fixed (same labels) | wall e- | 1 | 1706 | 0.798 | 0.812 | 7.067 | 6.071 | 0.821 | 0.077 | 0.780 |
| REF edge fixed (same labels) | wall e- | 2 | 1579 | 0.791 | 0.803 | 6.771 | 5.776 | 0.820 | 0.080 | 0.776 |
| REF edge fixed (same labels) | wall e- | 3 | 987 | 0.748 | 0.755 | 6.359 | 5.122 | 0.782 | 0.104 | 0.730 |
| REF edge fixed (same labels) | wall e- | all | 5465 | 0.781 | 0.793 | 6.747 | 5.683 | 0.810 | 0.088 | 0.765 |
| onion-checker-cont1 fixed | door e+ | 0 | 1254 | 0.285 | 0.284 | 4.574 | 1.505 | 0.284 | 0.702 | 0.270 |
| onion-checker-cont1 fixed | door e+ | 1 | 1723 | 0.102 | 0.095 | 2.003 | 0.480 | 0.062 | 0.887 | 0.059 |
| onion-checker-cont1 fixed | door e+ | 2 | 1493 | 0.079 | 0.074 | 1.737 | 0.565 | 0.050 | 0.872 | 0.046 |
| onion-checker-cont1 fixed | door e+ | 3 | 995 | 0.067 | 0.057 | 0.908 | 0.307 | 0.021 | 0.894 | 0.018 |
| onion-checker-cont1 fixed | door e+ | all | 5465 | 0.131 | 0.126 | 2.321 | 0.707 | 0.102 | 0.842 | 0.096 |
| onion-checker-cont1 fixed | wall e- | 0 | 1193 | 0.233 | 0.226 | 3.808 | 1.676 | 0.169 | 0.650 | 0.158 |
| onion-checker-cont1 fixed | wall e- | 1 | 1706 | 0.098 | 0.093 | 2.125 | 0.442 | 0.069 | 0.891 | 0.063 |
| onion-checker-cont1 fixed | wall e- | 2 | 1579 | 0.131 | 0.127 | 2.182 | 0.573 | 0.121 | 0.873 | 0.115 |
| onion-checker-cont1 fixed | wall e- | 3 | 987 | 0.067 | 0.059 | 1.084 | 0.208 | 0.049 | 0.939 | 0.049 |
| onion-checker-cont1 fixed | wall e- | all | 5465 | 0.131 | 0.126 | 2.321 | 0.707 | 0.102 | 0.842 | 0.096 |
| REF edge fixed (same labels) | door e+ | 0 | 1254 | 0.771 | 0.782 | 6.587 | 5.644 | 0.801 | 0.089 | 0.752 |
| REF edge fixed (same labels) | door e+ | 1 | 1723 | 0.802 | 0.813 | 7.089 | 5.812 | 0.828 | 0.084 | 0.786 |
| REF edge fixed (same labels) | door e+ | 2 | 1493 | 0.786 | 0.802 | 7.018 | 5.842 | 0.817 | 0.080 | 0.778 |
| REF edge fixed (same labels) | door e+ | 3 | 995 | 0.751 | 0.760 | 5.951 | 5.268 | 0.781 | 0.105 | 0.727 |
| REF edge fixed (same labels) | door e+ | all | 5465 | 0.781 | 0.793 | 6.747 | 5.683 | 0.810 | 0.088 | 0.765 |
| REF edge fixed (same labels) | wall e- | 0 | 1193 | 0.772 | 0.784 | 6.580 | 5.469 | 0.805 | 0.100 | 0.758 |
| REF edge fixed (same labels) | wall e- | 1 | 1706 | 0.798 | 0.812 | 7.067 | 6.071 | 0.821 | 0.077 | 0.780 |
| REF edge fixed (same labels) | wall e- | 2 | 1579 | 0.791 | 0.803 | 6.771 | 5.776 | 0.820 | 0.080 | 0.776 |
| REF edge fixed (same labels) | wall e- | 3 | 987 | 0.748 | 0.755 | 6.359 | 5.122 | 0.782 | 0.104 | 0.730 |
| REF edge fixed (same labels) | wall e- | all | 5465 | 0.781 | 0.793 | 6.747 | 5.683 | 0.810 | 0.088 | 0.765 |
| onion-checker-cont1 trainable | door e+ | 0 | 1254 | 0.184 | 0.171 | 3.708 | 0.844 | 0.167 | 0.811 | 0.159 |
| onion-checker-cont1 trainable | door e+ | 1 | 1723 | 0.055 | 0.049 | 1.080 | 0.223 | 0.031 | 0.942 | 0.028 |
| onion-checker-cont1 trainable | door e+ | 2 | 1493 | 0.092 | 0.090 | 1.567 | 0.637 | 0.054 | 0.854 | 0.052 |
| onion-checker-cont1 trainable | door e+ | 3 | 995 | 0.064 | 0.054 | 0.718 | 0.224 | 0.020 | 0.936 | 0.018 |
| onion-checker-cont1 trainable | door e+ | all | 5465 | 0.096 | 0.089 | 1.750 | 0.479 | 0.066 | 0.887 | 0.063 |
| onion-checker-cont1 trainable | wall e- | 0 | 1193 | 0.213 | 0.213 | 3.366 | 1.400 | 0.137 | 0.679 | 0.132 |
| onion-checker-cont1 trainable | wall e- | 1 | 1706 | 0.059 | 0.049 | 1.335 | 0.182 | 0.034 | 0.950 | 0.032 |
| onion-checker-cont1 trainable | wall e- | 2 | 1579 | 0.077 | 0.070 | 1.565 | 0.333 | 0.066 | 0.932 | 0.064 |
| onion-checker-cont1 trainable | wall e- | 3 | 987 | 0.049 | 0.040 | 0.811 | 0.112 | 0.036 | 0.956 | 0.032 |
| onion-checker-cont1 trainable | wall e- | all | 5465 | 0.096 | 0.089 | 1.750 | 0.479 | 0.066 | 0.887 | 0.063 |
| REF edge trainable (same labels) | door e+ | 0 | 1254 | 0.813 | 0.832 | 6.688 | 5.961 | 0.848 | 0.059 | 0.799 |
| REF edge trainable (same labels) | door e+ | 1 | 1723 | 0.831 | 0.852 | 7.007 | 6.196 | 0.867 | 0.054 | 0.825 |
| REF edge trainable (same labels) | door e+ | 2 | 1493 | 0.822 | 0.840 | 7.094 | 6.140 | 0.861 | 0.056 | 0.819 |
| REF edge trainable (same labels) | door e+ | 3 | 995 | 0.789 | 0.807 | 5.979 | 5.566 | 0.834 | 0.078 | 0.778 |
| REF edge trainable (same labels) | door e+ | all | 5465 | 0.817 | 0.836 | 6.771 | 6.012 | 0.855 | 0.060 | 0.809 |
| REF edge trainable (same labels) | wall e- | 0 | 1193 | 0.813 | 0.834 | 6.600 | 5.904 | 0.860 | 0.064 | 0.807 |
| REF edge trainable (same labels) | wall e- | 1 | 1706 | 0.829 | 0.845 | 7.135 | 6.266 | 0.859 | 0.059 | 0.820 |
| REF edge trainable (same labels) | wall e- | 2 | 1579 | 0.826 | 0.846 | 6.775 | 6.103 | 0.864 | 0.054 | 0.816 |
| REF edge trainable (same labels) | wall e- | 3 | 987 | 0.785 | 0.806 | 6.339 | 5.557 | 0.828 | 0.067 | 0.781 |
| REF edge trainable (same labels) | wall e- | all | 5465 | 0.817 | 0.836 | 6.771 | 6.012 | 0.855 | 0.060 | 0.809 |
| onion-full trainable | door e+ | 0 | 2093 | 0.600 | 0.594 | 5.836 | 3.989 | 0.573 | 0.263 | 0.473 |
| onion-full trainable | door e+ | 1 | 1390 | 0.368 | 0.360 | 3.705 | 2.050 | 0.315 | 0.458 | 0.255 |
| onion-full trainable | door e+ | 2 | 1368 | 0.348 | 0.341 | 3.140 | 2.641 | 0.271 | 0.466 | 0.225 |
| onion-full trainable | door e+ | 3 | 614 | 0.361 | 0.345 | 2.331 | 2.204 | 0.230 | 0.407 | 0.179 |
| onion-full trainable | door e+ | all | 5465 | 0.451 | 0.443 | 4.225 | 2.958 | 0.393 | 0.380 | 0.323 |
| onion-full trainable | wall e- | 0 | 2148 | 0.587 | 0.581 | 5.407 | 5.209 | 0.520 | 0.086 | 0.426 |
| onion-full trainable | wall e- | 1 | 1188 | 0.426 | 0.426 | 3.953 | 1.982 | 0.374 | 0.411 | 0.305 |
| onion-full trainable | wall e- | 2 | 1538 | 0.334 | 0.324 | 3.333 | 1.327 | 0.294 | 0.645 | 0.250 |
| onion-full trainable | wall e- | 3 | 591 | 0.308 | 0.288 | 2.800 | 0.982 | 0.232 | 0.694 | 0.173 |
| onion-full trainable | wall e- | all | 5465 | 0.451 | 0.443 | 4.225 | 2.958 | 0.393 | 0.380 | 0.323 |
| REF edge trainable (same labels) | door e+ | 0 | 2093 | 0.822 | 0.842 | 6.828 | 6.020 | 0.859 | 0.058 | 0.813 |
| REF edge trainable (same labels) | door e+ | 1 | 1390 | 0.815 | 0.836 | 6.589 | 6.031 | 0.854 | 0.057 | 0.809 |
| REF edge trainable (same labels) | door e+ | 2 | 1368 | 0.825 | 0.841 | 7.267 | 6.244 | 0.860 | 0.058 | 0.819 |
| REF edge trainable (same labels) | door e+ | 3 | 614 | 0.787 | 0.803 | 5.881 | 5.427 | 0.833 | 0.082 | 0.774 |
| REF edge trainable (same labels) | door e+ | all | 5465 | 0.817 | 0.836 | 6.771 | 6.012 | 0.855 | 0.060 | 0.809 |
| REF edge trainable (same labels) | wall e- | 0 | 2148 | 0.819 | 0.836 | 6.812 | 6.163 | 0.853 | 0.061 | 0.807 |
| REF edge trainable (same labels) | wall e- | 1 | 1188 | 0.817 | 0.838 | 6.960 | 5.796 | 0.860 | 0.069 | 0.818 |
| REF edge trainable (same labels) | wall e- | 2 | 1538 | 0.827 | 0.846 | 6.747 | 6.138 | 0.865 | 0.053 | 0.817 |
| REF edge trainable (same labels) | wall e- | 3 | 591 | 0.781 | 0.804 | 6.303 | 5.572 | 0.826 | 0.059 | 0.780 |
| REF edge trainable (same labels) | wall e- | all | 5465 | 0.817 | 0.836 | 6.771 | 6.012 | 0.855 | 0.060 | 0.809 |
| onion-full fixed | door e+ | 0 | 2093 | 0.612 | 0.615 | 5.802 | 3.915 | 0.586 | 0.247 | 0.492 |
| onion-full fixed | door e+ | 1 | 1390 | 0.468 | 0.450 | 4.336 | 2.759 | 0.422 | 0.377 | 0.356 |
| onion-full fixed | door e+ | 2 | 1368 | 0.327 | 0.325 | 2.991 | 2.546 | 0.222 | 0.450 | 0.178 |
| onion-full fixed | door e+ | 3 | 614 | 0.384 | 0.370 | 2.406 | 2.237 | 0.230 | 0.402 | 0.199 |
| onion-full fixed | door e+ | all | 5465 | 0.478 | 0.473 | 4.344 | 3.090 | 0.413 | 0.348 | 0.346 |
| onion-full fixed | wall e- | 0 | 2148 | 0.615 | 0.618 | 5.400 | 5.077 | 0.533 | 0.065 | 0.453 |
| onion-full fixed | wall e- | 1 | 1188 | 0.428 | 0.422 | 4.221 | 2.250 | 0.371 | 0.391 | 0.296 |
| onion-full fixed | wall e- | 2 | 1538 | 0.380 | 0.367 | 3.534 | 1.770 | 0.324 | 0.580 | 0.281 |
| onion-full fixed | wall e- | 3 | 591 | 0.340 | 0.325 | 2.861 | 0.993 | 0.291 | 0.689 | 0.223 |
| onion-full fixed | wall e- | all | 5465 | 0.478 | 0.473 | 4.344 | 3.090 | 0.413 | 0.348 | 0.346 |
| REF edge fixed (same labels) | door e+ | 0 | 2093 | 0.788 | 0.799 | 6.777 | 5.702 | 0.816 | 0.086 | 0.771 |
| REF edge fixed (same labels) | door e+ | 1 | 1390 | 0.777 | 0.788 | 6.671 | 5.597 | 0.808 | 0.089 | 0.761 |
| REF edge fixed (same labels) | door e+ | 2 | 1368 | 0.792 | 0.807 | 7.176 | 5.984 | 0.821 | 0.077 | 0.783 |
| REF edge fixed (same labels) | door e+ | 3 | 614 | 0.744 | 0.751 | 5.861 | 5.138 | 0.772 | 0.114 | 0.716 |
| REF edge fixed (same labels) | door e+ | all | 5465 | 0.781 | 0.793 | 6.747 | 5.683 | 0.810 | 0.088 | 0.765 |
| REF edge fixed (same labels) | wall e- | 0 | 2148 | 0.781 | 0.793 | 6.770 | 5.768 | 0.807 | 0.092 | 0.764 |
| REF edge fixed (same labels) | wall e- | 1 | 1188 | 0.786 | 0.800 | 6.915 | 5.652 | 0.819 | 0.089 | 0.774 |
| REF edge fixed (same labels) | wall e- | 2 | 1538 | 0.792 | 0.804 | 6.742 | 5.790 | 0.818 | 0.081 | 0.774 |
| REF edge fixed (same labels) | wall e- | 3 | 591 | 0.743 | 0.751 | 6.344 | 5.157 | 0.785 | 0.088 | 0.728 |
| REF edge fixed (same labels) | wall e- | all | 5465 | 0.781 | 0.793 | 6.747 | 5.683 | 0.810 | 0.088 | 0.765 |
| onion-full-cont1 trainable | door e+ | 0 | 2093 | 0.417 | 0.418 | 5.754 | 2.743 | 0.420 | 0.536 | 0.397 |
| onion-full-cont1 trainable | door e+ | 1 | 1390 | 0.172 | 0.169 | 2.720 | 1.004 | 0.144 | 0.763 | 0.131 |
| onion-full-cont1 trainable | door e+ | 2 | 1368 | 0.170 | 0.166 | 2.846 | 1.492 | 0.129 | 0.713 | 0.125 |
| onion-full-cont1 trainable | door e+ | 3 | 614 | 0.107 | 0.103 | 1.503 | 0.635 | 0.031 | 0.796 | 0.028 |
| onion-full-cont1 trainable | door e+ | all | 5465 | 0.258 | 0.256 | 3.777 | 1.751 | 0.233 | 0.667 | 0.220 |
| onion-full-cont1 trainable | wall e- | 0 | 2148 | 0.421 | 0.422 | 5.633 | 3.465 | 0.381 | 0.403 | 0.363 |
| onion-full-cont1 trainable | wall e- | 1 | 1188 | 0.164 | 0.157 | 2.680 | 0.779 | 0.133 | 0.806 | 0.125 |
| onion-full-cont1 trainable | wall e- | 2 | 1538 | 0.161 | 0.159 | 2.756 | 0.626 | 0.155 | 0.843 | 0.143 |
| onion-full-cont1 trainable | wall e- | 3 | 591 | 0.109 | 0.105 | 1.891 | 0.401 | 0.100 | 0.892 | 0.086 |
| onion-full-cont1 trainable | wall e- | all | 5465 | 0.258 | 0.256 | 3.777 | 1.751 | 0.233 | 0.667 | 0.220 |
| REF edge trainable (same labels) | door e+ | 0 | 2093 | 0.822 | 0.842 | 6.828 | 6.020 | 0.859 | 0.058 | 0.813 |
| REF edge trainable (same labels) | door e+ | 1 | 1390 | 0.815 | 0.836 | 6.589 | 6.031 | 0.854 | 0.057 | 0.809 |
| REF edge trainable (same labels) | door e+ | 2 | 1368 | 0.825 | 0.841 | 7.267 | 6.244 | 0.860 | 0.058 | 0.819 |
| REF edge trainable (same labels) | door e+ | 3 | 614 | 0.787 | 0.803 | 5.881 | 5.427 | 0.833 | 0.082 | 0.774 |
| REF edge trainable (same labels) | door e+ | all | 5465 | 0.817 | 0.836 | 6.771 | 6.012 | 0.855 | 0.060 | 0.809 |
| REF edge trainable (same labels) | wall e- | 0 | 2148 | 0.819 | 0.836 | 6.812 | 6.163 | 0.853 | 0.061 | 0.807 |
| REF edge trainable (same labels) | wall e- | 1 | 1188 | 0.817 | 0.838 | 6.960 | 5.796 | 0.860 | 0.069 | 0.818 |
| REF edge trainable (same labels) | wall e- | 2 | 1538 | 0.827 | 0.846 | 6.747 | 6.138 | 0.865 | 0.053 | 0.817 |
| REF edge trainable (same labels) | wall e- | 3 | 591 | 0.781 | 0.804 | 6.303 | 5.572 | 0.826 | 0.059 | 0.780 |
| REF edge trainable (same labels) | wall e- | all | 5465 | 0.817 | 0.836 | 6.771 | 6.012 | 0.855 | 0.060 | 0.809 |
| onion-full-cont1 fixed | door e+ | 0 | 2093 | 0.501 | 0.499 | 6.680 | 3.334 | 0.507 | 0.438 | 0.484 |
| onion-full-cont1 fixed | door e+ | 1 | 1390 | 0.251 | 0.244 | 4.013 | 1.652 | 0.220 | 0.639 | 0.209 |
| onion-full-cont1 fixed | door e+ | 2 | 1368 | 0.235 | 0.230 | 3.260 | 1.908 | 0.184 | 0.632 | 0.175 |
| onion-full-cont1 fixed | door e+ | 3 | 614 | 0.213 | 0.210 | 2.523 | 1.470 | 0.119 | 0.647 | 0.104 |
| onion-full-cont1 fixed | door e+ | all | 5465 | 0.339 | 0.334 | 4.678 | 2.340 | 0.310 | 0.561 | 0.294 |
| onion-full-cont1 fixed | wall e- | 0 | 2148 | 0.549 | 0.550 | 6.751 | 4.565 | 0.507 | 0.197 | 0.480 |
| onion-full-cont1 fixed | wall e- | 1 | 1188 | 0.206 | 0.199 | 3.589 | 0.883 | 0.177 | 0.783 | 0.165 |
| onion-full-cont1 fixed | wall e- | 2 | 1538 | 0.203 | 0.193 | 3.228 | 0.955 | 0.187 | 0.806 | 0.179 |
| onion-full-cont1 fixed | wall e- | 3 | 591 | 0.193 | 0.188 | 3.109 | 0.787 | 0.179 | 0.802 | 0.178 |
| onion-full-cont1 fixed | wall e- | all | 5465 | 0.339 | 0.334 | 4.678 | 2.340 | 0.310 | 0.561 | 0.294 |
| REF edge fixed (same labels) | door e+ | 0 | 2093 | 0.788 | 0.799 | 6.777 | 5.702 | 0.816 | 0.086 | 0.771 |
| REF edge fixed (same labels) | door e+ | 1 | 1390 | 0.777 | 0.788 | 6.671 | 5.597 | 0.808 | 0.089 | 0.761 |
| REF edge fixed (same labels) | door e+ | 2 | 1368 | 0.792 | 0.807 | 7.176 | 5.984 | 0.821 | 0.077 | 0.783 |
| REF edge fixed (same labels) | door e+ | 3 | 614 | 0.744 | 0.751 | 5.861 | 5.138 | 0.772 | 0.114 | 0.716 |
| REF edge fixed (same labels) | door e+ | all | 5465 | 0.781 | 0.793 | 6.747 | 5.683 | 0.810 | 0.088 | 0.765 |
| REF edge fixed (same labels) | wall e- | 0 | 2148 | 0.781 | 0.793 | 6.770 | 5.768 | 0.807 | 0.092 | 0.764 |
| REF edge fixed (same labels) | wall e- | 1 | 1188 | 0.786 | 0.800 | 6.915 | 5.652 | 0.819 | 0.089 | 0.774 |
| REF edge fixed (same labels) | wall e- | 2 | 1538 | 0.792 | 0.804 | 6.742 | 5.790 | 0.818 | 0.081 | 0.774 |
| REF edge fixed (same labels) | wall e- | 3 | 591 | 0.743 | 0.751 | 6.344 | 5.157 | 0.785 | 0.088 | 0.728 |
| REF edge fixed (same labels) | wall e- | all | 5465 | 0.781 | 0.793 | 6.747 | 5.683 | 0.810 | 0.088 | 0.765 |
