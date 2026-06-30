# Representation geometry — verdicts

## tier1
Z/12 collision: cycles SHARE a cyclic subspace (small principal angles) -> geometry follows the abstract Z/12 group, not semantics. months/color_wheel: theta=(0.731 [0.917,1.26],0.979 [1.16,1.51])deg, phase=40 [-40,40.2]deg; months/moon_phases: theta=(0.934 [1.06,1.45],1.01 [1.27,1.67])deg, phase=0.0013 [-0.199,0.229]deg; color_wheel/moon_phases: theta=(0.915 [1.04,1.47],1.01 [1.26,1.68])deg, phase=-40 [-40.2,40.1]deg. Z/4 (seasons/directions: theta2=1.67 [2,2.69]deg).
_figure_: `/Users/kaushikreddy/Projects/oracle-encoding-project/oracle-encodings/overnight_run/figures/tier1_cycles.png`

## tier2
Harmonic nesting: Seasons LIE IN the month plane (theta=(0.963 [1.18,1.73],1.33 [1.61,2.14])deg); coarse-graining dir(season)~mean(dir(months)) cosine=1 [0.999,0.999]; month centroids' 1st-harmonic energy fraction=0.999 [0.999,0.999] (seasons = fundamental Fourier mode of months). Base-10 in base-100: bucket centroid ~ mean of its unit members, cosine=1 [1,1] (multiscale magnitude coding).
_figure_: `/Users/kaushikreddy/Projects/oracle-encoding-project/oracle-encodings/overnight_run/figures/tier2_nesting.png`

## tier3
Abstract magnitude axis: a single shared axis explains 1 [1,1] of the variance across ['numbers', 'costliness', 'physical_size', 'duration'] (mean pairwise cosine 1 [0.999,1]). Cross-domain Spearman transfer from numbers: costliness=0.998 [0.997,0.999], physical=0.998 [0.996,0.998], duration=0.998 [0.997,0.999] (reused magnitude code). Spacing: numbers:linear(R2lin=1.00,R2log=0.93); costliness:log(R2lin=0.81,R2log=1.00); physical_size:log(R2lin=0.86,R2log=1.00); duration:log(R2lin=0.77,R2log=1.00). Moon illumination loads on the shared axis (|cos|=0.999 [0.997,0.999]).
_figure_: `/Users/kaushikreddy/Projects/oracle-encoding-project/oracle-encodings/overnight_run/figures/tier3_magnitude.png`

## tier4
World map: PCA layout of continent/place centroids Procrustes-aligns to true lat/long with disparity=4.37e-06 [3.69e-06,1.51e-05] (0=perfect metric map). Compass shares the map's frame: N-S vs latitude |cos|=0.98 [0.979,0.981], E-W vs longitude |cos|=0.998 [0.997,0.998].
_figure_: `/Users/kaushikreddy/Projects/oracle-encoding-project/oracle-encodings/overnight_run/figures/tier4_worldmap.png`

## tier5
Antipodal structure: indoors vs outdoors: cosine=-1 [-1,-1], angle=179 [179,179]deg -> ONE axis (antipodal, ~ -1) | lovingness vs harmfulness: cosine=0.0002 [-0.00173,0.00219], angle=90 [89.9,90.1]deg -> ORTHOGONAL (two independent features). 1-D check (top-PC var frac): harmfulness top-PC=0.995 [0.995,0.996]; indoors top-PC=0.98 [0.978,0.982]; lovingness top-PC=0.995 [0.994,0.995]; outdoors top-PC=0.98 [0.978,0.982].
_figure_: `/Users/kaushikreddy/Projects/oracle-encoding-project/oracle-encodings/overnight_run/figures/tier5_antipodal.png`
