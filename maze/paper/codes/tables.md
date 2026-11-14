| model | world | p(SP) 1 ep | solved 1 ep | p(SP) final | solved final | final step |
|---|---|---|---|---|---|---|
| axdeg fixed | true | 0.979 | 0.861 | 0.997 | 0.985 | 31212 |
| axdeg fixed | phantom | 0.913 | 0.521 | 0.919 | 0.575 | 31212 |
| axdeg1h fixed | true | 0.976 | 0.839 | 0.996 | 0.978 | 31212 |
| axdeg1h fixed | phantom | 0.832 | 0.169 | 0.796 | 0.087 | 31212 |
| axdeg1h trainable | true | 0.972 | 0.797 | 0.997 | 0.982 | 31212 |
| axdeg1h trainable | phantom | 0.882 | 0.356 | 0.845 | 0.227 | 31212 |
| base (no graft) | true | 0.974 | 0.831 | 0.996 | 0.973 | 31212 |
| edge fixed | true | 0.976 | 0.844 | 0.996 | 0.981 | 31212 |
| edge fixed | phantom | 0.908 | 0.471 | 0.915 | 0.534 | 31212 |
| ego fixed | true | 0.954 | 0.709 | 0.989 | 0.941 | 31212 |
| ego fixed | phantom | 0.910 | 0.478 | 0.926 | 0.556 | 31212 |
| ego trainable | true | 0.966 | 0.777 | 0.991 | 0.948 | 31212 |
| ego trainable | phantom | 0.928 | 0.606 | 0.933 | 0.610 | 31212 |
| REF base (5 seeds, intact prompt) | true | 0.971 | 0.795 | nan | nan | 7812 |
| REF fixed (5 seeds, intact prompt) | phantom | 0.908 | 0.480 | nan | nan | 7812 |
| REF fixed (5 seeds, intact prompt) | true | 0.975 | 0.838 | nan | nan | 7812 |
| REF trainable (5 seeds, intact prompt) | phantom | 0.913 | 0.519 | nan | nan | 7812 |
| REF trainable (5 seeds, intact prompt) | true | 0.974 | 0.828 | nan | nan | 7812 |

| model | site | abs walls @ path tokens | abs walls @ PATH_START | local N/E/S/W @ path tokens |
|---|---|---|---|---|
| axdeg fixed | L0in | 1.000 | 1.000 | 0.718 |
| axdeg fixed | L1 | 1.000 | 1.000 | 0.717 |
| axdeg fixed | L2 | 1.000 | 1.000 | 0.748 |
| axdeg fixed | L3 | 0.999 | 1.000 | 0.812 |
| axdeg fixed | L4 | 0.991 | 1.000 | 0.846 |
| axdeg fixed | L5 | 0.977 | 1.000 | 0.859 |
| axdeg fixed | L6 | 0.957 | 1.000 | 0.859 |
| axdeg fixed | L7 | 0.935 | 1.000 | 0.853 |
| axdeg fixed | L8 | 0.913 | 1.000 | 0.849 |
| axdeg fixed | L9 | 0.896 | 1.000 | 0.845 |
| axdeg fixed | L10 | 0.884 | 1.000 | 0.836 |
| axdeg fixed | L11 | 0.877 | 1.000 | 0.834 |
| axdeg fixed | L12 | 0.870 | 0.999 | 0.831 |
| axdeg1h fixed | L0in | 1.000 | 1.000 | 0.716 |
| axdeg1h fixed | L1 | 1.000 | 1.000 | 0.714 |
| axdeg1h fixed | L2 | 0.999 | 1.000 | 0.745 |
| axdeg1h fixed | L3 | 0.989 | 1.000 | 0.809 |
| axdeg1h fixed | L4 | 0.966 | 0.995 | 0.832 |
| axdeg1h fixed | L5 | 0.935 | 0.982 | 0.850 |
| axdeg1h fixed | L6 | 0.909 | 0.963 | 0.855 |
| axdeg1h fixed | L7 | 0.886 | 0.949 | 0.853 |
| axdeg1h fixed | L8 | 0.863 | 0.934 | 0.851 |
| axdeg1h fixed | L9 | 0.846 | 0.916 | 0.838 |
| axdeg1h fixed | L10 | 0.835 | 0.903 | 0.836 |
| axdeg1h fixed | L11 | 0.824 | 0.893 | 0.830 |
| axdeg1h fixed | L12 | 0.817 | 0.888 | 0.822 |
| axdeg1h trainable | L0in | 1.000 | 1.000 | 0.716 |
| axdeg1h trainable | L1 | 1.000 | 1.000 | 0.724 |
| axdeg1h trainable | L2 | 1.000 | 1.000 | 0.785 |
| axdeg1h trainable | L3 | 0.995 | 1.000 | 0.831 |
| axdeg1h trainable | L4 | 0.985 | 1.000 | 0.864 |
| axdeg1h trainable | L5 | 0.970 | 1.000 | 0.878 |
| axdeg1h trainable | L6 | 0.951 | 0.999 | 0.878 |
| axdeg1h trainable | L7 | 0.930 | 0.998 | 0.875 |
| axdeg1h trainable | L8 | 0.911 | 0.997 | 0.866 |
| axdeg1h trainable | L9 | 0.897 | 0.993 | 0.855 |
| axdeg1h trainable | L10 | 0.884 | 0.989 | 0.848 |
| axdeg1h trainable | L11 | 0.876 | 0.985 | 0.841 |
| axdeg1h trainable | L12 | 0.869 | 0.981 | 0.833 |
| base (no graft) | L0in | 0.610 | 0.602 | 0.690 |
| base (no graft) | L1 | 0.725 | 0.781 | 0.713 |
| base (no graft) | L2 | 0.756 | 0.922 | 0.828 |
| base (no graft) | L3 | 0.801 | 0.986 | 0.882 |
| base (no graft) | L4 | 0.782 | 0.994 | 0.880 |
| base (no graft) | L5 | 0.776 | 0.993 | 0.891 |
| base (no graft) | L6 | 0.764 | 0.990 | 0.883 |
| base (no graft) | L7 | 0.748 | 0.979 | 0.871 |
| base (no graft) | L8 | 0.739 | 0.967 | 0.866 |
| base (no graft) | L9 | 0.730 | 0.957 | 0.851 |
| base (no graft) | L10 | 0.721 | 0.934 | 0.841 |
| base (no graft) | L11 | 0.716 | 0.915 | 0.836 |
| base (no graft) | L12 | 0.712 | 0.894 | 0.832 |
| edge fixed | L0in | 1.000 | 1.000 | 0.718 |
| edge fixed | L1 | 1.000 | 1.000 | 0.719 |
| edge fixed | L2 | 1.000 | 1.000 | 0.741 |
| edge fixed | L3 | 1.000 | 1.000 | 0.792 |
| edge fixed | L4 | 0.999 | 1.000 | 0.829 |
| edge fixed | L5 | 0.996 | 1.000 | 0.852 |
| edge fixed | L6 | 0.989 | 1.000 | 0.859 |
| edge fixed | L7 | 0.975 | 1.000 | 0.849 |
| edge fixed | L8 | 0.961 | 1.000 | 0.843 |
| edge fixed | L9 | 0.946 | 1.000 | 0.834 |
| edge fixed | L10 | 0.932 | 1.000 | 0.829 |
| edge fixed | L11 | 0.923 | 1.000 | 0.820 |
| edge fixed | L12 | 0.915 | 0.999 | 0.813 |
| ego fixed | L0in | 0.682 | 0.655 | 1.000 |
| ego fixed | L1 | 0.715 | 0.715 | 1.000 |
| ego fixed | L2 | 0.709 | 0.716 | 1.000 |
| ego fixed | L3 | 0.713 | 0.727 | 1.000 |
| ego fixed | L4 | 0.717 | 0.744 | 1.000 |
| ego fixed | L5 | 0.719 | 0.754 | 1.000 |
| ego fixed | L6 | 0.716 | 0.750 | 1.000 |
| ego fixed | L7 | 0.710 | 0.742 | 1.000 |
| ego fixed | L8 | 0.704 | 0.748 | 1.000 |
| ego fixed | L9 | 0.698 | 0.744 | 0.999 |
| ego fixed | L10 | 0.694 | 0.740 | 0.998 |
| ego fixed | L11 | 0.691 | 0.735 | 0.997 |
| ego fixed | L12 | 0.689 | 0.733 | 0.996 |
| ego trainable | L0in | 0.682 | 0.656 | 1.000 |
| ego trainable | L1 | 0.736 | 0.751 | 1.000 |
| ego trainable | L2 | 0.717 | 0.728 | 1.000 |
| ego trainable | L3 | 0.714 | 0.724 | 1.000 |
| ego trainable | L4 | 0.724 | 0.752 | 1.000 |
| ego trainable | L5 | 0.724 | 0.757 | 1.000 |
| ego trainable | L6 | 0.728 | 0.775 | 1.000 |
| ego trainable | L7 | 0.712 | 0.769 | 1.000 |
| ego trainable | L8 | 0.703 | 0.759 | 1.000 |
| ego trainable | L9 | 0.694 | 0.752 | 0.999 |
| ego trainable | L10 | 0.689 | 0.750 | 0.999 |
| ego trainable | L11 | 0.686 | 0.747 | 0.999 |
| ego trainable | L12 | 0.683 | 0.745 | 0.998 |

Intact prompt, standard set: solve rate on route-changed items (graft maze / prompt maze)

| model | n | same_maze | route_changed_graft | route_changed_prompt |
|---|---|---|---|---|
| axdeg fixed | 1 | 0.987 | 0.612 | 0.037 |
| axdeg1h fixed | 1 | 0.977 | 0.082 | 0.425 |
| axdeg1h trainable | 1 | 0.983 | 0.311 | 0.092 |
| edge fixed | 1 | 0.980 | 0.725 | 0.016 |
| ego fixed | 1 | 0.944 | 0.714 | 0.005 |
| ego trainable | 1 | 0.949 | 0.721 | 0.004 |
| REF edge fixed (5 seeds, 1 ep, adjlist_end) | 5 | 0.838 | 0.716 | 0.009 |
| REF edge trainable (5 seeds, 1 ep, adjlist_end) | 5 | 0.832 | 0.747 | 0.006 |

Intact prompt, one-swap set (DFS family): graft carries the edited maze, prompt shows the old one

| model | n | p_new | top1_new | gain_nats | used_door | crossed_wall | solved_new | prompt_only_p_new | both_p_new |
|---|---|---|---|---|---|---|---|---|---|
| axdeg fixed | 1 | 0.766 | 0.773 | 8.361 | 0.768 | 0.182 | 0.758 | 0.196 | 0.976 |
| axdeg1h fixed | 1 | 0.191 | 0.185 | 3.210 | 0.144 | 0.803 | 0.139 | 0.767 | 0.969 |
| axdeg1h trainable | 1 | 0.500 | 0.500 | 6.547 | 0.487 | 0.411 | 0.471 | 0.426 | 0.962 |
| edge fixed | 1 | 0.771 | 0.780 | 8.471 | 0.786 | 0.120 | 0.767 | 0.183 | 0.969 |
| ego fixed | 1 | 0.813 | 0.815 | 8.230 | 0.848 | 0.069 | 0.808 | 0.161 | 0.941 |
| ego trainable | 1 | 0.829 | 0.829 | 8.501 | 0.872 | 0.059 | 0.827 | 0.151 | 0.954 |
| REF edge fixed (5 seeds, 1 ep, adjlist_end) | 5 | 0.781 | 0.793 | 6.747 | 0.810 | 0.088 | 0.765 | 0.182 | 0.883 |
| REF edge trainable (5 seeds, 1 ep, adjlist_end) | 5 | 0.817 | 0.836 | 6.771 | 0.855 | 0.060 | 0.809 | 0.159 | 0.882 |

Inference-time ablations (true world, 2,000 mazes; prompt/generation split at the origin token)

| model | cond | p_sp | p_valid | p_invalid | solved | reached | clean | valid_move_frac |
|---|---|---|---|---|---|---|---|---|
| axdeg fixed | full | 0.997 | 0.003 | 0.000 | 0.985 | 0.999 | 0.985 | 0.999 |
| axdeg fixed | graft_off | 0.114 | 0.098 | 0.788 | 0.000 | 0.004 | 0.001 | 0.078 |
| axdeg fixed | off_gen | 0.285 | 0.070 | 0.645 | 0.005 | 0.068 | 0.005 | 0.865 |
| axdeg fixed | off_prompt | 0.393 | 0.170 | 0.437 | 0.000 | 0.006 | 0.001 | 0.158 |
| axdeg fixed | x0.5 | 0.937 | 0.028 | 0.035 | 0.550 | 0.945 | 0.550 | 0.933 |
| axdeg fixed | x2 | 0.235 | 0.156 | 0.609 | 0.007 | 0.032 | 0.042 | 0.314 |
| axdeg fixed | noise | 0.036 | 0.048 | 0.916 | 0.000 | 0.017 | 0.000 | 0.067 |
| axdeg fixed | prompt_blank | 0.886 | 0.067 | 0.047 | 0.372 | 0.936 | 0.380 | 0.898 |
| axdeg fixed | blank_off | 0.183 | 0.202 | 0.616 | 0.000 | 0.026 | 0.000 | 0.503 |
| axdeg fixed | shuffled | 0.647 | 0.095 | 0.258 | 0.093 | 0.980 | 0.093 | 0.666 |
| axdeg1h fixed | full | 0.996 | 0.003 | 0.000 | 0.978 | 0.998 | 0.979 | 0.998 |
| axdeg1h fixed | graft_off | 0.045 | 0.040 | 0.915 | 0.000 | 0.000 | 0.000 | 0.000 |
| axdeg1h fixed | off_gen | 0.204 | 0.039 | 0.758 | 0.001 | 0.053 | 0.001 | 0.903 |
| axdeg1h fixed | off_prompt | 0.323 | 0.153 | 0.524 | 0.000 | 0.000 | 0.000 | 0.000 |
| axdeg1h fixed | x0.5 | 0.828 | 0.113 | 0.059 | 0.273 | 0.386 | 0.274 | 0.881 |
| axdeg1h fixed | x2 | 0.220 | 0.182 | 0.598 | 0.002 | 0.026 | 0.004 | 0.272 |
| axdeg1h fixed | noise | 0.032 | 0.043 | 0.925 | 0.000 | 0.016 | 0.000 | 0.053 |
| axdeg1h fixed | prompt_blank | 0.696 | 0.094 | 0.210 | 0.140 | 0.850 | 0.143 | 0.760 |
| axdeg1h fixed | blank_off | 0.238 | 0.243 | 0.519 | 0.000 | 0.000 | 0.000 | 0.000 |
| axdeg1h fixed | shuffled | 0.916 | 0.043 | 0.041 | 0.411 | 0.973 | 0.411 | 0.902 |
| axdeg1h trainable | full | 0.997 | 0.003 | 0.000 | 0.982 | 0.996 | 0.982 | 0.998 |
| axdeg1h trainable | graft_off | 0.080 | 0.059 | 0.860 | 0.000 | 0.009 | 0.000 | 0.162 |
| axdeg1h trainable | off_gen | 0.295 | 0.046 | 0.659 | 0.010 | 0.067 | 0.010 | 0.918 |
| axdeg1h trainable | off_prompt | 0.291 | 0.204 | 0.505 | 0.000 | 0.011 | 0.002 | 0.214 |
| axdeg1h trainable | x0.5 | 0.726 | 0.036 | 0.238 | 0.173 | 0.231 | 0.173 | 0.730 |
| axdeg1h trainable | x2 | 0.330 | 0.208 | 0.462 | 0.007 | 0.058 | 0.031 | 0.604 |
| axdeg1h trainable | noise | 0.038 | 0.049 | 0.913 | 0.000 | 0.015 | 0.001 | 0.074 |
| axdeg1h trainable | prompt_blank | 0.519 | 0.211 | 0.270 | 0.082 | 0.372 | 0.086 | 0.840 |
| axdeg1h trainable | blank_off | 0.025 | 0.035 | 0.940 | 0.000 | 0.035 | 0.000 | 0.166 |
| axdeg1h trainable | shuffled | 0.718 | 0.064 | 0.218 | 0.133 | 0.964 | 0.133 | 0.722 |
| base (no graft) | full | 0.996 | 0.004 | 0.000 | 0.973 | 0.994 | 0.974 | 0.998 |
| base (no graft) | prompt_blank | 0.024 | 0.035 | 0.940 | 0.000 | 0.018 | 0.000 | 0.113 |
| edge fixed | full | 0.996 | 0.003 | 0.000 | 0.981 | 0.999 | 0.981 | 0.999 |
| edge fixed | graft_off | 0.056 | 0.046 | 0.897 | 0.000 | 0.001 | 0.000 | 0.008 |
| edge fixed | off_gen | 0.200 | 0.039 | 0.761 | 0.001 | 0.045 | 0.001 | 0.939 |
| edge fixed | off_prompt | 0.381 | 0.221 | 0.397 | 0.000 | 0.001 | 0.000 | 0.011 |
| edge fixed | x0.5 | 0.948 | 0.024 | 0.027 | 0.628 | 0.909 | 0.628 | 0.952 |
| edge fixed | x2 | 0.195 | 0.171 | 0.634 | 0.002 | 0.029 | 0.047 | 0.372 |
| edge fixed | noise | 0.023 | 0.031 | 0.945 | 0.000 | 0.009 | 0.000 | 0.042 |
| edge fixed | prompt_blank | 0.359 | 0.234 | 0.406 | 0.001 | 0.017 | 0.003 | 0.534 |
| edge fixed | blank_off | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| edge fixed | shuffled | 0.614 | 0.090 | 0.297 | 0.083 | 0.977 | 0.083 | 0.636 |
| ego fixed | full | 0.989 | 0.010 | 0.001 | 0.941 | 0.992 | 0.944 | 0.996 |
| ego fixed | graft_off | 0.109 | 0.061 | 0.829 | 0.004 | 0.059 | 0.015 | 0.236 |
| ego fixed | off_gen | 0.231 | 0.057 | 0.712 | 0.016 | 0.074 | 0.057 | 0.697 |
| ego fixed | off_prompt | 0.784 | 0.093 | 0.123 | 0.080 | 0.553 | 0.113 | 0.720 |
| ego fixed | x0.5 | 0.944 | 0.045 | 0.012 | 0.675 | 0.793 | 0.818 | 0.981 |
| ego fixed | x2 | 0.647 | 0.306 | 0.047 | 0.013 | 0.029 | 0.321 | 0.958 |
| ego fixed | noise | 0.035 | 0.040 | 0.925 | 0.000 | 0.018 | 0.021 | 0.073 |
| ego fixed | prompt_blank | 0.827 | 0.153 | 0.020 | 0.249 | 0.347 | 0.496 | 0.963 |
| ego fixed | blank_off | 0.113 | 0.072 | 0.815 | 0.000 | 0.020 | 0.065 | 0.203 |
| ego fixed | shuffled | 0.594 | 0.087 | 0.319 | 0.073 | 0.956 | 0.073 | 0.624 |
| ego trainable | full | 0.991 | 0.008 | 0.000 | 0.948 | 0.988 | 0.951 | 0.996 |
| ego trainable | graft_off | 0.011 | 0.012 | 0.977 | 0.000 | 0.047 | 0.002 | 0.120 |
| ego trainable | off_gen | 0.133 | 0.009 | 0.858 | 0.000 | 0.051 | 0.001 | 0.967 |
| ego trainable | off_prompt | 0.852 | 0.028 | 0.120 | 0.103 | 0.774 | 0.106 | 0.659 |
| ego trainable | x0.5 | 0.971 | 0.014 | 0.015 | 0.811 | 0.921 | 0.819 | 0.983 |
| ego trainable | x2 | 0.773 | 0.214 | 0.013 | 0.018 | 0.031 | 0.395 | 0.980 |
| ego trainable | noise | 0.032 | 0.045 | 0.923 | 0.000 | 0.023 | 0.003 | 0.071 |
| ego trainable | prompt_blank | 0.844 | 0.140 | 0.016 | 0.271 | 0.412 | 0.429 | 0.959 |
| ego trainable | blank_off | 0.066 | 0.040 | 0.894 | 0.002 | 0.083 | 0.016 | 0.207 |
| ego trainable | shuffled | 0.586 | 0.088 | 0.327 | 0.073 | 0.930 | 0.073 | 0.620 |
