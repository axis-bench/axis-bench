# Simulation results: What Matters in Quality-Aware VLA Pretraining: A Benchmark and Empirical Study

Every number on this page is recomputed by qvla paper tables (T0) from the per-seed rollout files of the 2 cells in qvla-v1-results. A cell is one fine-tuned checkpoint served one way and evaluated on the 24 tasks with the same 50 layouts each (1,200 trials). SR is the mean task success (%), and its SE is the task-stratified binomial SE, 100 sqrt(Σ_t p_t(1 − p_t)/50)/24 over the per-task success rates p_t. A relative value is 100(SR/SR_BC − 1) from unrounded rates, where SR_BC is the VLA's 5K plain BC (plain FT, untagged inference), and its SE is the SE scaled by 100/SR_BC. Fig. 1 and Table I instead compare two cells of one training seed and divide by that seed's own plain BC, 100(SR(A) − SR(B))/SR_BC; the SE of a Fig. 1 per-seed value is that of the difference, 100 sqrt(SE_A² + SE_B²)/SR_BC. A difference in percentage points (pp) is SR(A) − SR(B).

## Headline gains (Fig. 1)

**Fig. 1. Mean relative gain of approach-aware episode conditioning over plain BC (Table I, first comparison)**

| VLA | group | bar | SR (%) | relative (%) | SE (%) | note | cells |
|---|---|---|---|---|---|---|---|
| π0.5 | Episode cond. − BC | seed 42 | 83.33 | +66.7 | 47.1 |  | cfg_ph_ep\|cond; bc\|bare |
| π0.5 | Episode cond. − BC | seed 43 | — | — | — |  | cfgpt43\|cond; bc43\|bare |
| π0.5 | Episode cond. − BC | seed 44 | — | — | — |  | cfgpt44\|cond; bc44\|bare |
| π0.5 | Episode cond. − BC | mean over seeds | — | — | — | the value Fig. 1 prints |  |
| SmolVLA | Episode cond. − BC | seed 0 | — | — | — |  | svla_cfgph\|cond; svla_bc\|bare |
| SmolVLA | Episode cond. − BC | seed 1 | — | — | — |  | svla_cfgph_sd1\|cond; svla_bc_sd1\|bare |
| SmolVLA | Episode cond. − BC | seed 2 | — | — | — |  | svla_cfgph_sd2\|cond; svla_bc_sd2\|bare |
| SmolVLA | Episode cond. − BC | mean over seeds | — | — | — | the value Fig. 1 prints |  |
| X-VLA | Episode cond. − BC | seed 0 | — | — | — |  | xvla_cfgph\|cond; xvla_bc\|bare |
| X-VLA | Episode cond. − BC | seed 1 | — | — | — |  | xvla_cfgph_sd1\|cond; xvla_bc_sd1\|bare |
| X-VLA | Episode cond. − BC | seed 2 | — | — | — |  | xvla_cfgph_sd2\|cond; xvla_bc_sd2\|bare |
| X-VLA | Episode cond. − BC | mean over seeds | — | — | — | the value Fig. 1 prints |  |

## IV.B From reward changes to quality labels: quality filtering versus random selection

**Fig. 4(c). Random selection versus approach-aware quality filtering at the same half-data budget, relative to each VLA's full-data plain BC**

| VLA | group | bar | SR (%) | relative (%) | SE (%) | note | cells |
|---|---|---|---|---|---|---|---|
| SmolVLA | random selection | episode | — | — | — |  | svla_rand_ep\|bare |
| SmolVLA | random selection | chunk | — | — | — |  | svla_rand_row\|bare |
| SmolVLA | approach-aware filtering | episode | — | — | — |  | svla_drop_ph_ep\|bare |
| SmolVLA | approach-aware filtering | chunk | — | — | — |  | svla_drop_ph_row\|bare |
| X-VLA | random selection | episode | — | — | — |  | xvla_rand_ep\|bare |
| X-VLA | random selection | chunk | — | — | — |  | xvla_rand_row\|bare |
| X-VLA | approach-aware filtering | episode | — | — | — |  | xvla_drop_ph_ep\|bare |
| X-VLA | approach-aware filtering | chunk | — | — | — |  | xvla_drop_ph_row\|bare |

**In-text numbers**

| claim | value | cells / per comparison |
|---|---|---|
| SmolVLA: approach-aware episode filtering − random episode selection (pp) | — | svla_drop_ph_ep\|bare, svla_rand_ep\|bare |
| SmolVLA: approach-aware chunk filtering − random chunk selection (pp) | — | svla_drop_ph_row\|bare, svla_rand_row\|bare |
| SmolVLA: object-centric chunk filtering − random chunk selection (pp) | — | svla_drop_v2_row\|bare, svla_rand_row\|bare |
| X-VLA: approach-aware episode filtering − random episode selection (pp) | — | xvla_drop_ph_ep\|bare, xvla_rand_ep\|bare |
| X-VLA: approach-aware chunk filtering − random chunk selection (pp) | — | xvla_drop_ph_row\|bare, xvla_rand_row\|bare |
| X-VLA: object-centric chunk filtering − random chunk selection (pp) | — | xvla_drop_v2_row\|bare, xvla_rand_row\|bare |

## V.B Cross-VLA gains from episode conditioning

**Fig. 5. Weighting, top-50% filtering and conditioning on 24 tasks, relative to plain BC (plain FT, untagged inference)**

| VLA | group | bar | SR (%) | relative (%) | SE (%) | note | cells |
|---|---|---|---|---|---|---|---|
| π0.5 | plain BC | reference | 50.00 | 0.0 | 38.5 | plain FT / untagged inference | bc\|bare |
| π0.5 | FT-only control | no AXIS pretraining | — | — | — | matched FT-only control | droidinit\|bare |
| π0.5 | weighting | object-centric / episode | — | — | — |  | awr_v2_ep\|bare |
| π0.5 | weighting | object-centric / chunk | — | — | — |  | awr_v2_row\|bare |
| π0.5 | weighting | approach-aware / episode | — | — | — |  | awr_ph_ep\|bare |
| π0.5 | weighting | approach-aware / chunk | — | — | — |  | awr_ph_row\|bare |
| π0.5 | filtering | object-centric / episode | — | — | — |  | drop_v2_ep\|bare |
| π0.5 | filtering | object-centric / chunk | — | — | — |  | drop_v2_row\|bare |
| π0.5 | filtering | approach-aware / episode | — | — | — |  | drop_ph_ep\|bare |
| π0.5 | filtering | approach-aware / chunk | — | — | — |  | drop_ph_row\|bare |
| π0.5 | conditioning | object-centric / episode | — | — | — |  | cfg_v2_ep\|cond |
| π0.5 | conditioning | object-centric / chunk | — | — | — |  | cfg_v2_row\|cond |
| π0.5 | conditioning | approach-aware / episode | 83.33 | +66.7 | 27.2 |  | cfg_ph_ep\|cond |
| π0.5 | conditioning | approach-aware / chunk | — | — | — |  | cfg_ph_row\|cond |
| SmolVLA | plain BC | reference | — | — | — | plain FT / untagged inference | svla_bc\|bare |
| SmolVLA | FT-only control | no AXIS pretraining | — | — | — | matched FT-only control | svla_base_rand\|bare |
| SmolVLA | weighting | object-centric / episode | — | — | — |  | svla_awr_v2_ep\|bare |
| SmolVLA | weighting | object-centric / chunk | — | — | — |  | svla_awr_v2_row\|bare |
| SmolVLA | weighting | approach-aware / episode | — | — | — |  | svla_awr_ph_ep\|bare |
| SmolVLA | weighting | approach-aware / chunk | — | — | — |  | svla_awr_ph_row\|bare |
| SmolVLA | filtering | object-centric / episode | — | — | — |  | svla_drop_v2_ep\|bare |
| SmolVLA | filtering | object-centric / chunk | — | — | — |  | svla_drop_v2_row\|bare |
| SmolVLA | filtering | approach-aware / episode | — | — | — |  | svla_drop_ph_ep\|bare |
| SmolVLA | filtering | approach-aware / chunk | — | — | — |  | svla_drop_ph_row\|bare |
| SmolVLA | conditioning | object-centric / episode | — | — | — |  | svla_cfg_v2_ep\|cond |
| SmolVLA | conditioning | object-centric / chunk | — | — | — |  | svla_cfg_v2_row\|cond |
| SmolVLA | conditioning | approach-aware / episode | — | — | — |  | svla_cfgph\|cond |
| SmolVLA | conditioning | approach-aware / chunk | — | — | — |  | svla_cfg_ph_row\|cond |
| X-VLA | plain BC | reference | — | — | — | plain FT / untagged inference | xvla_bc\|bare |
| X-VLA | FT-only control | no AXIS pretraining | — | — | — | matched FT-only control | xvla_base\|bare |
| X-VLA | weighting | object-centric / episode | — | — | — |  | xvla_awr_v2_ep\|bare |
| X-VLA | weighting | object-centric / chunk | — | — | — |  | xvla_awr_v2_row\|bare |
| X-VLA | weighting | approach-aware / episode | — | — | — |  | xvla_awr_ph_ep\|bare |
| X-VLA | weighting | approach-aware / chunk | — | — | — |  | xvla_awr_ph_row\|bare |
| X-VLA | filtering | object-centric / episode | — | — | — |  | xvla_drop_v2_ep\|bare |
| X-VLA | filtering | object-centric / chunk | — | — | — |  | xvla_drop_v2_row\|bare |
| X-VLA | filtering | approach-aware / episode | — | — | — |  | xvla_drop_ph_ep\|bare |
| X-VLA | filtering | approach-aware / chunk | — | — | — |  | xvla_drop_ph_row\|bare |
| X-VLA | conditioning | object-centric / episode | — | — | — |  | xvla_cfg_v2_ep\|cond |
| X-VLA | conditioning | object-centric / chunk | — | — | — |  | xvla_cfg_v2_row\|cond |
| X-VLA | conditioning | approach-aware / episode | — | — | — |  | xvla_cfgph\|cond |
| X-VLA | conditioning | approach-aware / chunk | — | — | — |  | xvla_cfg_ph_row\|cond |

**In-text numbers**

| claim | value | cells / per comparison |
|---|---|---|
| π0.5: approach-aware episode conditioning relative to plain BC (%) | +66.67 | cfg_ph_ep\|cond, bc\|bare |
| SmolVLA: approach-aware episode conditioning relative to plain BC (%) | — | svla_cfgph\|cond, svla_bc\|bare |
| X-VLA: approach-aware episode conditioning relative to plain BC (%) | — | xvla_cfgph\|cond, xvla_bc\|bare |
| π0.5: mean success rate (%) of each method over the four reward-granularity settings | weighting —, filtering —, conditioning — |  |
| SmolVLA: mean success rate (%) of each method over the four reward-granularity settings | weighting —, filtering —, conditioning — |  |
| X-VLA: mean success rate (%) of each method over the four reward-granularity settings | weighting —, filtering —, conditioning — |  |
| X-VLA: filtering − conditioning in each matched reward-granularity setting | 0 positive, 0 equal, 0 negative, 4 missing | object-centric / episode —; object-centric / chunk —; approach-aware / episode —; approach-aware / chunk — |
| π0.5: plain-BC success rate (%) | 50.00 | bc\|bare |
| π0.5: matched FT-only control success rate (%) | — | droidinit\|bare |
| SmolVLA: plain-BC success rate (%) | — | svla_bc\|bare |
| SmolVLA: matched FT-only control success rate (%) | — | svla_base_rand\|bare |
| X-VLA: plain-BC success rate (%) | — | xvla_bc\|bare |
| X-VLA: matched FT-only control success rate (%) | — | xvla_base\|bare |

## V.C Coupling between label granularity and learning method

**In-text numbers**

| claim | value | cells / per comparison |
|---|---|---|
| episode − chunk conditioning in each matched VLA-reward setting | 0 positive, 0 equal, 0 negative, 6 missing | π0.5 object-centric —; π0.5 approach-aware —; SmolVLA object-centric —; SmolVLA approach-aware —; X-VLA object-centric —; X-VLA approach-aware — |
| chunk − episode filtering in each matched VLA-reward setting | 0 positive, 0 equal, 0 negative, 6 missing | π0.5 object-centric —; π0.5 approach-aware —; SmolVLA object-centric —; SmolVLA approach-aware —; X-VLA object-centric —; X-VLA approach-aware — |
| π0.5: approach-aware episode − chunk conditioning (pp) | — | cfg_ph_ep\|cond, cfg_ph_row\|cond |
| SmolVLA: approach-aware episode − chunk conditioning (pp) | — | svla_cfgph\|cond, svla_cfg_ph_row\|cond |
| X-VLA: approach-aware episode − chunk conditioning (pp) | — | xvla_cfgph\|cond, xvla_cfg_ph_row\|cond |
| π0.5: object-centric episode − chunk conditioning (pp) | — | cfg_v2_ep\|cond, cfg_v2_row\|cond |
| X-VLA: object-centric chunk − episode filtering (pp) | — | xvla_drop_v2_row\|bare, xvla_drop_v2_ep\|bare |
| X-VLA: random chunk − random episode selection (pp) | — | xvla_rand_row\|bare, xvla_rand_ep\|bare |
| object-centric weighting: episode − chunk, per VLA | 0 positive, 0 equal, 0 negative, 3 missing | π0.5 —; SmolVLA —; X-VLA — |
| approach-aware weighting: episode − chunk, per VLA | 0 positive, 0 equal, 0 negative, 3 missing | π0.5 —; SmolVLA —; X-VLA — |
| approach-aware chunk weighting − plain BC, per VLA (pp) | 0 positive, 0 equal, 0 negative, 3 missing | π0.5 —; SmolVLA —; X-VLA — |

## V.D Retaining quality tags during fine-tuning

**Fig. 7. Conditioning stage and inference context, approach-aware episode labels, relative to plain BC / plain FT / untagged inference**

| VLA | group | bar | SR (%) | relative (%) | SE (%) | note | cells |
|---|---|---|---|---|---|---|---|
| π0.5 | BC pretraining / plain FT | untagged inference | 50.00 | 0.0 | 38.5 | the plain-BC reference band; native protocol | bc\|bare |
| π0.5 | BC pretraining / plain FT | tag-5 inference | — | — | — | Sec. V.D only; not a Fig. 7 bar | bc\|cond |
| π0.5 | BC pretraining / tagged FT | untagged inference | — | — | — |  | bct\|bare |
| π0.5 | BC pretraining / tagged FT | tag-5 inference | — | — | — | native protocol | bct\|cond |
| π0.5 | conditioning / plain FT | untagged inference | — | — | — | native protocol | cfgnp\|bare |
| π0.5 | conditioning / plain FT | tag-5 inference | — | — | — |  | cfgnp\|cond |
| π0.5 | conditioning / tagged FT | untagged inference | — | — | — |  | cfg_ph_ep\|bare |
| π0.5 | conditioning / tagged FT | tag-5 inference | 83.33 | +66.7 | 27.2 | native protocol | cfg_ph_ep\|cond |
| SmolVLA | BC pretraining / plain FT | untagged inference | — | — | — | the plain-BC reference band; native protocol | svla_bc\|bare |
| SmolVLA | BC pretraining / plain FT | tag-5 inference | — | — | — | Sec. V.D only; not a Fig. 7 bar | svla_bc\|cond |
| SmolVLA | BC pretraining / tagged FT | untagged inference | — | — | — |  | svla_bct\|bare |
| SmolVLA | BC pretraining / tagged FT | tag-5 inference | — | — | — | native protocol | svla_bct\|cond |
| SmolVLA | conditioning / plain FT | untagged inference | — | — | — | native protocol | svla_cfgnp\|bare |
| SmolVLA | conditioning / plain FT | tag-5 inference | — | — | — |  | svla_cfgnp\|cond |
| SmolVLA | conditioning / tagged FT | untagged inference | — | — | — |  | svla_cfgph\|bare |
| SmolVLA | conditioning / tagged FT | tag-5 inference | — | — | — | native protocol | svla_cfgph\|cond |
| X-VLA | BC pretraining / plain FT | untagged inference | — | — | — | the plain-BC reference band; native protocol | xvla_bc\|bare |
| X-VLA | BC pretraining / tagged FT | untagged inference | — | — | — |  | xvla_bct\|bare |
| X-VLA | BC pretraining / tagged FT | tag-5 inference | — | — | — | native protocol | xvla_bct\|cond |
| X-VLA | conditioning / plain FT | untagged inference | — | — | — | native protocol | xvla_cfgnp\|bare |
| X-VLA | conditioning / plain FT | tag-5 inference | — | — | — |  | xvla_cfgnp\|cond |
| X-VLA | conditioning / tagged FT | untagged inference | — | — | — |  | xvla_cfgph\|bare |
| X-VLA | conditioning / tagged FT | tag-5 inference | — | — | — | native protocol | xvla_cfgph\|cond |

**In-text numbers**

| claim | value | cells / per comparison |
|---|---|---|
| π0.5: tagged − plain FT after conditioning, tag-5 inference (pp) | — | cfg_ph_ep\|cond, cfgnp\|cond |
| π0.5: tagged − plain FT after conditioning, untagged inference (pp) | — | cfg_ph_ep\|bare, cfgnp\|bare |
| SmolVLA: tagged − plain FT after conditioning, tag-5 inference (pp) | — | svla_cfgph\|cond, svla_cfgnp\|cond |
| SmolVLA: tagged − plain FT after conditioning, untagged inference (pp) | — | svla_cfgph\|bare, svla_cfgnp\|bare |
| X-VLA: tagged − plain FT after conditioning, tag-5 inference (pp) | — | xvla_cfgph\|cond, xvla_cfgnp\|cond |
| X-VLA: tagged − plain FT after conditioning, untagged inference (pp) | — | xvla_cfgph\|bare, xvla_cfgnp\|bare |
| π0.5: adding tag 5 at inference after conditioning + plain FT (pp) | — | cfgnp\|cond, cfgnp\|bare |
| π0.5: adding tag 5 at inference after conditioning + tagged FT (pp) | — | cfg_ph_ep\|cond, cfg_ph_ep\|bare |
| SmolVLA: adding tag 5 at inference after conditioning + plain FT (pp) | — | svla_cfgnp\|cond, svla_cfgnp\|bare |
| SmolVLA: adding tag 5 at inference after conditioning + tagged FT (pp) | — | svla_cfgph\|cond, svla_cfgph\|bare |
| X-VLA: adding tag 5 at inference after conditioning + plain FT (pp) | — | xvla_cfgnp\|cond, xvla_cfgnp\|bare |
| X-VLA: adding tag 5 at inference after conditioning + tagged FT (pp) | — | xvla_cfgph\|cond, xvla_cfgph\|bare |
| π0.5: after plain-BC pretraining, tagged FT + tag-5 inference − plain FT + untagged inference (pp) | — | bct\|cond, bc\|bare |
| SmolVLA: after plain-BC pretraining, tagged FT + tag-5 inference − plain FT + untagged inference (pp) | — | svla_bct\|cond, svla_bc\|bare |
| X-VLA: after plain-BC pretraining, tagged FT + tag-5 inference − plain FT + untagged inference (pp) | — | xvla_bct\|cond, xvla_bc\|bare |
| π0.5: after plain-BC pretraining and plain FT, tag-5 − untagged inference (pp) | — | bc\|cond, bc\|bare |
| SmolVLA: after plain-BC pretraining and plain FT, tag-5 − untagged inference (pp) | — | svla_bc\|cond, svla_bc\|bare |

## V.E Scaling with dataset size

**Fig. 6. Scaling with pretraining data size, relative to each VLA's fixed 5K plain BC**

| VLA | group | bar | SR (%) | relative (%) | SE (%) | note | cells |
|---|---|---|---|---|---|---|---|
| π0.5 | plain BC | 1K | — | — | — | plain FT / untagged inference | s1000bc\|bare |
| π0.5 | plain BC | 2.5K | — | — | — | plain FT / untagged inference | s2500bc\|bare |
| π0.5 | plain BC | 5K | 50.00 | 0.0 | 38.5 | plain FT / untagged inference | bc\|bare |
| π0.5 | conditioning | 1K | — | — | — | tagged FT / tag-5 inference | s1000cfg\|cond |
| π0.5 | conditioning | 2.5K | — | — | — | tagged FT / tag-5 inference | s2500cfg\|cond |
| π0.5 | conditioning | 5K | 83.33 | +66.7 | 27.2 | tagged FT / tag-5 inference | cfg_ph_ep\|cond |
| SmolVLA | plain BC | 1K | — | — | — | plain FT / untagged inference | svla_s1000bc\|bare |
| SmolVLA | plain BC | 2.5K | — | — | — | plain FT / untagged inference | svla_s2500bc\|bare |
| SmolVLA | plain BC | 5K | — | — | — | plain FT / untagged inference | svla_bc\|bare |
| SmolVLA | conditioning | 1K | — | — | — | tagged FT / tag-5 inference | svla_s1000cfg\|cond |
| SmolVLA | conditioning | 2.5K | — | — | — | tagged FT / tag-5 inference | svla_s2500cfg\|cond |
| SmolVLA | conditioning | 5K | — | — | — | tagged FT / tag-5 inference | svla_cfgph\|cond |
| X-VLA | plain BC | 1K | — | — | — | plain FT / untagged inference | xvla_s1000bc\|bare |
| X-VLA | plain BC | 2.5K | — | — | — | plain FT / untagged inference | xvla_s2500bc\|bare |
| X-VLA | plain BC | 5K | — | — | — | plain FT / untagged inference | xvla_bc\|bare |
| X-VLA | conditioning | 1K | — | — | — | tagged FT / tag-5 inference | xvla_s1000cfg\|cond |
| X-VLA | conditioning | 2.5K | — | — | — | tagged FT / tag-5 inference | xvla_s2500cfg\|cond |
| X-VLA | conditioning | 5K | — | — | — | tagged FT / tag-5 inference | xvla_cfgph\|cond |

**In-text numbers**

| claim | value | cells / per comparison |
|---|---|---|
| π0.5: conditioning − plain BC at matched pretraining size | 1 positive, 0 equal, 0 negative, 2 missing | 1K —; 2.5K —; 5K +33.33 |
| π0.5: plain BC at 1K (%) | — | s1000bc\|bare |
| π0.5: conditioning at 1K (%) | — | s1000cfg\|cond |
| π0.5: plain BC at 2.5K (%) | — | s2500bc\|bare |
| π0.5: conditioning at 2.5K (%) | — | s2500cfg\|cond |
| π0.5: plain BC at 5K (%) | 50.00 | bc\|bare |
| π0.5: conditioning at 5K (%) | 83.33 | cfg_ph_ep\|cond |
| SmolVLA: conditioning − plain BC at matched pretraining size | 0 positive, 0 equal, 0 negative, 3 missing | 1K —; 2.5K —; 5K — |
| SmolVLA: plain BC at 1K (%) | — | svla_s1000bc\|bare |
| SmolVLA: conditioning at 1K (%) | — | svla_s1000cfg\|cond |
| SmolVLA: plain BC at 2.5K (%) | — | svla_s2500bc\|bare |
| SmolVLA: conditioning at 2.5K (%) | — | svla_s2500cfg\|cond |
| SmolVLA: plain BC at 5K (%) | — | svla_bc\|bare |
| SmolVLA: conditioning at 5K (%) | — | svla_cfgph\|cond |
| X-VLA: conditioning − plain BC at matched pretraining size | 0 positive, 0 equal, 0 negative, 3 missing | 1K —; 2.5K —; 5K — |
| X-VLA: plain BC at 1K (%) | — | xvla_s1000bc\|bare |
| X-VLA: conditioning at 1K (%) | — | xvla_s1000cfg\|cond |
| X-VLA: plain BC at 2.5K (%) | — | xvla_s2500bc\|bare |
| X-VLA: conditioning at 2.5K (%) | — | xvla_s2500cfg\|cond |
| X-VLA: plain BC at 5K (%) | — | xvla_bc\|bare |
| X-VLA: conditioning at 5K (%) | — | xvla_cfgph\|cond |

## V.F Training-seed replication

**Table I. Training-seed replication. Entries are mean ± sample SD of BC-normalized success-rate differences (%) over three seeds per VLA, using approach-aware reward**

| Comparison | π0.5 | SmolVLA | X-VLA |
|---|---:|---:|---:|
| Episode cond. − BC | — | — | — |
| Cond.: episode − chunk | — | — | — |
| Filter: chunk − episode | — | — | — |
| Tagged − plain: untagged | — | — | — |
| Tagged − plain: tag 5 | — | — | — |

**Table I per training seed (%)**

| VLA | comparison | seed | value |
|---|---|---|---|
| π0.5 | Episode cond. − BC | 42 | +66.67 |
| π0.5 | Episode cond. − BC | 43 | — |
| π0.5 | Episode cond. − BC | 44 | — |
| SmolVLA | Episode cond. − BC | 0 | — |
| SmolVLA | Episode cond. − BC | 1 | — |
| SmolVLA | Episode cond. − BC | 2 | — |
| X-VLA | Episode cond. − BC | 0 | — |
| X-VLA | Episode cond. − BC | 1 | — |
| X-VLA | Episode cond. − BC | 2 | — |
| π0.5 | Cond.: episode − chunk | 42 | — |
| π0.5 | Cond.: episode − chunk | 43 | — |
| π0.5 | Cond.: episode − chunk | 44 | — |
| SmolVLA | Cond.: episode − chunk | 0 | — |
| SmolVLA | Cond.: episode − chunk | 1 | — |
| SmolVLA | Cond.: episode − chunk | 2 | — |
| X-VLA | Cond.: episode − chunk | 0 | — |
| X-VLA | Cond.: episode − chunk | 1 | — |
| X-VLA | Cond.: episode − chunk | 2 | — |
| π0.5 | Filter: chunk − episode | 42 | — |
| π0.5 | Filter: chunk − episode | 43 | — |
| π0.5 | Filter: chunk − episode | 44 | — |
| SmolVLA | Filter: chunk − episode | 0 | — |
| SmolVLA | Filter: chunk − episode | 1 | — |
| SmolVLA | Filter: chunk − episode | 2 | — |
| X-VLA | Filter: chunk − episode | 0 | — |
| X-VLA | Filter: chunk − episode | 1 | — |
| X-VLA | Filter: chunk − episode | 2 | — |
| π0.5 | Tagged − plain: untagged | 42 | — |
| π0.5 | Tagged − plain: untagged | 43 | — |
| π0.5 | Tagged − plain: untagged | 44 | — |
| SmolVLA | Tagged − plain: untagged | 0 | — |
| SmolVLA | Tagged − plain: untagged | 1 | — |
| SmolVLA | Tagged − plain: untagged | 2 | — |
| X-VLA | Tagged − plain: untagged | 0 | — |
| X-VLA | Tagged − plain: untagged | 1 | — |
| X-VLA | Tagged − plain: untagged | 2 | — |
| π0.5 | Tagged − plain: tag 5 | 42 | — |
| π0.5 | Tagged − plain: tag 5 | 43 | — |
| π0.5 | Tagged − plain: tag 5 | 44 | — |
| SmolVLA | Tagged − plain: tag 5 | 0 | — |
| SmolVLA | Tagged − plain: tag 5 | 1 | — |
| SmolVLA | Tagged − plain: tag 5 | 2 | — |
| X-VLA | Tagged − plain: tag 5 | 0 | — |
| X-VLA | Tagged − plain: tag 5 | 1 | — |
| X-VLA | Tagged − plain: tag 5 | 2 | — |

A training seed is a replicate of the pretraining run, keyed by its pretraining seed; each cell's fine-tuning seed is recorded in cells.json. Cells whose fine-tuning seed differs from the replicate's seed:

| VLA | seed | role | cell | pretraining seed | fine-tuning seed |
|---|---|---|---|---|---|

**In-text numbers**

| claim | value | cells / per comparison |
|---|---|---|
| seed-level episode conditioning − BC comparisons (nine VLA-seed combinations) | 1 positive, 0 equal, 0 negative, 8 missing |  |
| conditioning, episode − chunk: equal average over the nine VLA-seed combinations (%) | — |  |
| filtering, chunk − episode: equal average over the nine VLA-seed combinations (%) | — |  |
| per-VLA means of the Table I comparison Cond.: episode − chunk | 0 positive, 0 equal, 0 negative, 3 missing |  |
| per-VLA means of the Table I comparison Filter: chunk − episode | 0 positive, 0 equal, 0 negative, 3 missing |  |
| VLA-inference comparisons of tagged − plain FT (means over seeds) | 0 positive, 0 equal, 0 negative, 6 missing |  |
| seed-level tagged − plain FT differences | 0 positive, 0 equal, 0 negative, 18 missing |  |
| SmolVLA: tag-5 − untagged inference after conditioning + tagged FT, per training seed | 0 positive, 0 equal, 0 negative, 3 missing | seed 0 —; seed 1 —; seed 2 — |
| X-VLA: tag-5 − untagged inference after conditioning + tagged FT, per training seed | 0 positive, 0 equal, 0 negative, 3 missing | seed 0 —; seed 1 —; seed 2 — |
