# Validation runs

Produced with `sh scripts/run_validation.sh data results/raw` and `scripts/compare_with_paper.py`, seed 0, 10 runs of random search and the GA, 10 splits per labelled-set size. Weighted F1 on the test documents, mean over the 10 splits. Paper values are from Tables 5–7 of Coutinho, Rezende and Rossi (2019). Total CPU time for both presets was about 8 minutes on an Apple laptop.

Data: CSTR is the ARFF copy from the ICMC-USP benchmark; Re0, Re1, Tr45 and WAP are the CLUTO matrices fetched by `scripts/download_data.py`.

## Preset `notebook` (default)

| Collection | n/class | preset | Grid paper | Grid ours | Random paper | Random ours | GA best paper | GA best ours (max / by-val) | GA avg paper | GA avg ours (sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| CSTR | 1 | notebook | 0.4089 | 0.4167 | 0.4072 | 0.4081 | 0.4120 | 0.4210 / 0.4171 | 0.4089 | 0.4120 (0.0071) |
| CSTR | 10 | notebook | 0.7445 | 0.7339 | 0.7434 | 0.7342 | 0.7461 | 0.7328 / 0.7276 | 0.7457 | 0.7258 (0.0082) |
| CSTR | 20 | notebook | 0.7851 | 0.7119 | 0.7860 | 0.7187 | 0.7859 | 0.7179 / 0.7158 | 0.7859 | 0.7158 (0.0021) |
| re0 | 1 | notebook | 0.2900 | 0.2422 | 0.2915 | 0.2421 | 0.2916 | 0.2434 / 0.2421 | 0.2916 | 0.2423 (0.0004) |
| re0 | 10 | notebook | 0.6125 | 0.6010 | 0.6120 | 0.5987 | 0.6151 | 0.5942 / 0.5901 | 0.6151 | 0.5900 (0.0024) |
| re1 | 1 | notebook | 0.3177 | 0.3105 | 0.3183 | 0.3139 | 0.3206 | 0.3125 / 0.3125 | 0.3203 | 0.3118 (0.0006) |
| re1 | 10 | notebook | 0.6256 | 0.6179 | 0.6265 | 0.6146 | 0.6321 | 0.6176 / 0.6155 | 0.6319 | 0.6146 (0.0021) |
| tr45 | 1 | notebook | 0.4312 | 0.3510 | 0.4355 | 0.3434 | 0.4392 | 0.3514 / 0.3514 | 0.4392 | 0.3436 (0.0051) |
| tr45 | 10 | notebook | 0.4110 | 0.5097 | 0.4109 | 0.5083 | 0.4112 | 0.5124 / 0.5095 | 0.4112 | 0.5094 (0.0026) |
| wap | 1 | notebook | 0.3327 | 0.3193 | 0.3329 | 0.3177 | 0.3342 | 0.3230 / 0.3230 | 0.3341 | 0.3194 (0.0026) |

## Preset `clean`

| Collection | n/class | preset | Grid paper | Grid ours | Random paper | Random ours | GA best paper | GA best ours (max / by-val) | GA avg paper | GA avg ours (sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| CSTR | 1 | clean | 0.4089 | 0.4339 | 0.4072 | 0.4338 | 0.4120 | 0.4342 / 0.4342 | 0.4089 | 0.4178 (0.0089) |
| CSTR | 10 | clean | 0.7445 | 0.7208 | 0.7434 | 0.7296 | 0.7461 | 0.7393 / 0.7203 | 0.7457 | 0.7251 (0.0070) |
| CSTR | 20 | clean | 0.7851 | 0.7238 | 0.7860 | 0.7212 | 0.7859 | 0.7281 / 0.7264 | 0.7859 | 0.7183 (0.0073) |
| re0 | 1 | clean | 0.2900 | 0.2377 | 0.2915 | 0.2411 | 0.2916 | 0.2421 / 0.2421 | 0.2916 | 0.2416 (0.0008) |
| re0 | 10 | clean | 0.6125 | 0.5955 | 0.6120 | 0.5960 | 0.6151 | 0.6012 / 0.5993 | 0.6151 | 0.5923 (0.0053) |
| re1 | 1 | clean | 0.3177 | 0.3116 | 0.3183 | 0.3132 | 0.3206 | 0.3135 / 0.3122 | 0.3203 | 0.3123 (0.0008) |
| re1 | 10 | clean | 0.6256 | 0.6139 | 0.6265 | 0.6117 | 0.6321 | 0.6186 / 0.6186 | 0.6319 | 0.6105 (0.0029) |
| tr45 | 1 | clean | 0.4312 | 0.3414 | 0.4355 | 0.3434 | 0.4392 | 0.3413 / 0.3412 | 0.4392 | 0.3404 (0.0010) |
| tr45 | 10 | clean | 0.4110 | 0.5081 | 0.4109 | 0.5060 | 0.4112 | 0.5082 / 0.5069 | 0.4112 | 0.5062 (0.0015) |
| wap | 1 | clean | 0.3327 | 0.3332 | 0.3329 | 0.3214 | 0.3342 | 0.3301 / 0.3301 | 0.3341 | 0.3253 (0.0040) |

## How much the splits alone move the result

Test F1 of a fixed hyper-parameter triple (the grid-search choice for seed 0, `notebook` preset), averaged over 10 splits, for five different split seeds:

| Collection | n/class | (SF, k, α) | seeds 0–4 | sd of one split |
|---|---|---|---|---|
| Tr45 | 1 | (4, 1, 0.3) | 0.351, 0.333, 0.403, 0.427, 0.421 | 0.163 |
| Re0 | 1 | (5, 1, 0.3) | 0.242, 0.233, 0.266, 0.319, 0.243 | 0.096 |
| CSTR | 20 | (4, 57, 0.3) | 0.712, 0.710, 0.726, 0.726, 0.757 | 0.039 |
| Tr45 | 10 | (1, 1, 0.7) | 0.510, 0.439, 0.315, 0.384, 0.415 | 0.252 |
