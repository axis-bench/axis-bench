# Simulation results: What Matters in Quality-Aware VLA Pretraining: A Benchmark and Empirical Study

Every number on this page is recomputed by qvla paper tables (T0) from the per-seed rollout files of the 123 cells in qvla-v1-results. A cell is one fine-tuned checkpoint served one way and evaluated on the 24 tasks with the same 50 layouts each (1,200 trials). SR is the mean task success (%), and its SE is the task-stratified binomial SE, 100 sqrt(Σ_t p_t(1 − p_t)/50)/24 over the per-task success rates p_t. A relative value is 100(SR/SR_BC − 1) from unrounded rates, where SR_BC is the VLA's 5K plain BC (plain FT, untagged inference), and its SE is the SE scaled by 100/SR_BC. Fig. 1 and Table I instead compare two cells of one training seed and divide by that seed's own plain BC, 100(SR(A) − SR(B))/SR_BC; the SE of a Fig. 1 per-seed value is that of the difference, 100 sqrt(SE_A² + SE_B²)/SR_BC. A difference in percentage points (pp) is SR(A) − SR(B).

## Headline gains (Fig. 1)

**Fig. 1. Mean relative gain of approach-aware episode conditioning over plain BC (Table I, first comparison)**

| VLA | group | bar | SR (%) | relative (%) | SE (%) | note | cells |
|---|---|---|---|---|---|---|---|
| π0.5 | Episode cond. − BC | seed 42 | 58.00 | +1.0 | 2.6 |  | cfg_ph_ep\|cond; bc\|bare |
| π0.5 | Episode cond. − BC | seed 43 | 51.33 | 0.0 | 3.0 |  | cfgpt43\|cond; bc43\|bare |
| π0.5 | Episode cond. − BC | seed 44 | 57.58 | +10.6 | 2.9 |  | cfgpt44\|cond; bc44\|bare |
| π0.5 | Episode cond. − BC | mean over seeds | — | +3.9 | — | the value Fig. 1 prints |  |
| SmolVLA | Episode cond. − BC | seed 0 | 8.83 | +73.8 | 18.8 |  | svla_cfgph\|cond; svla_bc\|bare |
| SmolVLA | Episode cond. − BC | seed 1 | 6.75 | +11.0 | 14.7 |  | svla_cfgph_sd1\|cond; svla_bc_sd1\|bare |
| SmolVLA | Episode cond. − BC | seed 2 | 5.42 | −14.5 | 14.2 |  | svla_cfgph_sd2\|cond; svla_bc_sd2\|bare |
| SmolVLA | Episode cond. − BC | mean over seeds | — | +23.4 | — | the value Fig. 1 prints |  |
| X-VLA | Episode cond. − BC | seed 0 | 27.83 | +8.1 | 5.6 |  | xvla_cfgph\|cond; xvla_bc\|bare |
| X-VLA | Episode cond. − BC | seed 1 | 31.25 | +9.3 | 5.1 |  | xvla_cfgph_sd1\|cond; xvla_bc_sd1\|bare |
| X-VLA | Episode cond. − BC | seed 2 | 29.58 | −0.6 | 5.2 |  | xvla_cfgph_sd2\|cond; xvla_bc_sd2\|bare |
| X-VLA | Episode cond. − BC | mean over seeds | — | +5.6 | — | the value Fig. 1 prints |  |

## IV.B From reward changes to quality labels: quality filtering versus random selection

**Fig. 4(c). Random selection versus approach-aware quality filtering at the same half-data budget, relative to each VLA's full-data plain BC**

| VLA | group | bar | SR (%) | relative (%) | SE (%) | note | cells |
|---|---|---|---|---|---|---|---|
| SmolVLA | random selection | episode | 4.00 | −21.3 | 10.6 |  | svla_rand_ep\|bare |
| SmolVLA | random selection | chunk | 3.92 | −23.0 | 10.6 |  | svla_rand_row\|bare |
| SmolVLA | approach-aware filtering | episode | 5.08 | 0.0 | 11.7 |  | svla_drop_ph_ep\|bare |
| SmolVLA | approach-aware filtering | chunk | 7.00 | +37.7 | 13.1 |  | svla_drop_ph_row\|bare |
| X-VLA | random selection | episode | 23.00 | −10.7 | 3.8 |  | xvla_rand_ep\|bare |
| X-VLA | random selection | chunk | 26.58 | +3.2 | 4.2 |  | xvla_rand_row\|bare |
| X-VLA | approach-aware filtering | episode | 28.75 | +11.7 | 4.1 |  | xvla_drop_ph_ep\|bare |
| X-VLA | approach-aware filtering | chunk | 29.17 | +13.3 | 3.8 |  | xvla_drop_ph_row\|bare |

**In-text numbers**

| claim | value | cells / per comparison |
|---|---|---|
| SmolVLA: approach-aware episode filtering − random episode selection (pp) | +1.08 | svla_drop_ph_ep\|bare, svla_rand_ep\|bare |
| SmolVLA: approach-aware chunk filtering − random chunk selection (pp) | +3.08 | svla_drop_ph_row\|bare, svla_rand_row\|bare |
| SmolVLA: object-centric chunk filtering − random chunk selection (pp) | +3.17 | svla_drop_v2_row\|bare, svla_rand_row\|bare |
| X-VLA: approach-aware episode filtering − random episode selection (pp) | +5.75 | xvla_drop_ph_ep\|bare, xvla_rand_ep\|bare |
| X-VLA: approach-aware chunk filtering − random chunk selection (pp) | +2.58 | xvla_drop_ph_row\|bare, xvla_rand_row\|bare |
| X-VLA: object-centric chunk filtering − random chunk selection (pp) | −3.00 | xvla_drop_v2_row\|bare, xvla_rand_row\|bare |

## V.B Cross-VLA gains from episode conditioning

**Fig. 5. Weighting, top-50% filtering and conditioning on 24 tasks, relative to plain BC (plain FT, untagged inference)**

| VLA | group | bar | SR (%) | relative (%) | SE (%) | note | cells |
|---|---|---|---|---|---|---|---|
| π0.5 | plain BC | reference | 57.42 | 0.0 | 1.9 | plain FT / untagged inference | bc\|bare |
| π0.5 | FT-only control | no AXIS pretraining | 8.83 | −84.6 | 1.3 | matched FT-only control | droidinit\|bare |
| π0.5 | weighting | object-centric / episode | 56.67 | −1.3 | 1.9 |  | awr_v2_ep\|bare |
| π0.5 | weighting | object-centric / chunk | 52.33 | −8.9 | 1.8 |  | awr_v2_row\|bare |
| π0.5 | weighting | approach-aware / episode | 54.67 | −4.8 | 1.8 |  | awr_ph_ep\|bare |
| π0.5 | weighting | approach-aware / chunk | 58.08 | +1.2 | 1.8 |  | awr_ph_row\|bare |
| π0.5 | filtering | object-centric / episode | 49.75 | −13.4 | 2.0 |  | drop_v2_ep\|bare |
| π0.5 | filtering | object-centric / chunk | 53.83 | −6.2 | 1.8 |  | drop_v2_row\|bare |
| π0.5 | filtering | approach-aware / episode | 50.33 | −12.3 | 1.8 |  | drop_ph_ep\|bare |
| π0.5 | filtering | approach-aware / chunk | 50.33 | −12.3 | 1.8 |  | drop_ph_row\|bare |
| π0.5 | conditioning | object-centric / episode | 52.92 | −7.8 | 1.9 |  | cfg_v2_ep\|cond |
| π0.5 | conditioning | object-centric / chunk | 56.83 | −1.0 | 1.9 |  | cfg_v2_row\|cond |
| π0.5 | conditioning | approach-aware / episode | 58.00 | +1.0 | 1.9 |  | cfg_ph_ep\|cond |
| π0.5 | conditioning | approach-aware / chunk | 54.50 | −5.1 | 1.8 |  | cfg_ph_row\|cond |
| SmolVLA | plain BC | reference | 5.08 | 0.0 | 11.7 | plain FT / untagged inference | svla_bc\|bare |
| SmolVLA | FT-only control | no AXIS pretraining | 2.50 | −50.8 | 8.0 | matched FT-only control | svla_base_rand\|bare |
| SmolVLA | weighting | object-centric / episode | 5.83 | +14.8 | 12.3 |  | svla_awr_v2_ep\|bare |
| SmolVLA | weighting | object-centric / chunk | 5.33 | +4.9 | 11.9 |  | svla_awr_v2_row\|bare |
| SmolVLA | weighting | approach-aware / episode | 6.83 | +34.4 | 13.2 |  | svla_awr_ph_ep\|bare |
| SmolVLA | weighting | approach-aware / chunk | 5.58 | +9.8 | 12.4 |  | svla_awr_ph_row\|bare |
| SmolVLA | filtering | object-centric / episode | 3.75 | −26.2 | 10.3 |  | svla_drop_v2_ep\|bare |
| SmolVLA | filtering | object-centric / chunk | 7.08 | +39.3 | 12.1 |  | svla_drop_v2_row\|bare |
| SmolVLA | filtering | approach-aware / episode | 5.08 | 0.0 | 11.7 |  | svla_drop_ph_ep\|bare |
| SmolVLA | filtering | approach-aware / chunk | 7.00 | +37.7 | 13.1 |  | svla_drop_ph_row\|bare |
| SmolVLA | conditioning | object-centric / episode | 9.17 | +80.3 | 14.8 |  | svla_cfg_v2_ep\|cond |
| SmolVLA | conditioning | object-centric / chunk | 6.58 | +29.5 | 12.8 |  | svla_cfg_v2_row\|cond |
| SmolVLA | conditioning | approach-aware / episode | 8.83 | +73.8 | 14.7 |  | svla_cfgph\|cond |
| SmolVLA | conditioning | approach-aware / chunk | 5.92 | +16.4 | 12.0 |  | svla_cfg_ph_row\|cond |
| X-VLA | plain BC | reference | 25.75 | 0.0 | 3.9 | plain FT / untagged inference | xvla_bc\|bare |
| X-VLA | FT-only control | no AXIS pretraining | 6.17 | −76.1 | 2.5 | matched FT-only control | xvla_base\|bare |
| X-VLA | weighting | object-centric / episode | 28.42 | +10.4 | 4.1 |  | xvla_awr_v2_ep\|bare |
| X-VLA | weighting | object-centric / chunk | 26.50 | +2.9 | 4.0 |  | xvla_awr_v2_row\|bare |
| X-VLA | weighting | approach-aware / episode | 25.75 | 0.0 | 4.0 |  | xvla_awr_ph_ep\|bare |
| X-VLA | weighting | approach-aware / chunk | 28.00 | +8.7 | 4.2 |  | xvla_awr_ph_row\|bare |
| X-VLA | filtering | object-centric / episode | 28.25 | +9.7 | 4.0 |  | xvla_drop_v2_ep\|bare |
| X-VLA | filtering | object-centric / chunk | 23.58 | −8.4 | 4.0 |  | xvla_drop_v2_row\|bare |
| X-VLA | filtering | approach-aware / episode | 28.75 | +11.7 | 4.1 |  | xvla_drop_ph_ep\|bare |
| X-VLA | filtering | approach-aware / chunk | 29.17 | +13.3 | 3.8 |  | xvla_drop_ph_row\|bare |
| X-VLA | conditioning | object-centric / episode | 23.50 | −8.7 | 4.0 |  | xvla_cfg_v2_ep\|cond |
| X-VLA | conditioning | object-centric / chunk | 20.42 | −20.7 | 3.8 |  | xvla_cfg_v2_row\|cond |
| X-VLA | conditioning | approach-aware / episode | 27.83 | +8.1 | 4.1 |  | xvla_cfgph\|cond |
| X-VLA | conditioning | approach-aware / chunk | 25.83 | +0.3 | 3.8 |  | xvla_cfg_ph_row\|cond |

**In-text numbers**

| claim | value | cells / per comparison |
|---|---|---|
| π0.5: approach-aware episode conditioning relative to plain BC (%) | +1.02 | cfg_ph_ep\|cond, bc\|bare |
| SmolVLA: approach-aware episode conditioning relative to plain BC (%) | +73.77 | svla_cfgph\|cond, svla_bc\|bare |
| X-VLA: approach-aware episode conditioning relative to plain BC (%) | +8.09 | xvla_cfgph\|cond, xvla_bc\|bare |
| π0.5: mean success rate (%) of each method over the four reward-granularity settings | weighting 55.44, filtering 51.06, conditioning 55.56 |  |
| SmolVLA: mean success rate (%) of each method over the four reward-granularity settings | weighting 5.90, filtering 5.73, conditioning 7.62 |  |
| X-VLA: mean success rate (%) of each method over the four reward-granularity settings | weighting 27.17, filtering 27.44, conditioning 24.40 |  |
| X-VLA: filtering − conditioning in each matched reward-granularity setting | 4 positive, 0 equal, 0 negative | object-centric / episode +4.75; object-centric / chunk +3.17; approach-aware / episode +0.92; approach-aware / chunk +3.33 |
| π0.5: plain-BC success rate (%) | 57.42 | bc\|bare |
| π0.5: matched FT-only control success rate (%) | 8.83 | droidinit\|bare |
| SmolVLA: plain-BC success rate (%) | 5.08 | svla_bc\|bare |
| SmolVLA: matched FT-only control success rate (%) | 2.50 | svla_base_rand\|bare |
| X-VLA: plain-BC success rate (%) | 25.75 | xvla_bc\|bare |
| X-VLA: matched FT-only control success rate (%) | 6.17 | xvla_base\|bare |

## V.C Coupling between label granularity and learning method

**In-text numbers**

| claim | value | cells / per comparison |
|---|---|---|
| episode − chunk conditioning in each matched VLA-reward setting | 5 positive, 0 equal, 1 negative | π0.5 object-centric −3.92; π0.5 approach-aware +3.50; SmolVLA object-centric +2.58; SmolVLA approach-aware +2.92; X-VLA object-centric +3.08; X-VLA approach-aware +2.00 |
| chunk − episode filtering in each matched VLA-reward setting | 4 positive, 1 equal, 1 negative | π0.5 object-centric +4.08; π0.5 approach-aware 0.00; SmolVLA object-centric +3.33; SmolVLA approach-aware +1.92; X-VLA object-centric −4.67; X-VLA approach-aware +0.42 |
| π0.5: approach-aware episode − chunk conditioning (pp) | +3.50 | cfg_ph_ep\|cond, cfg_ph_row\|cond |
| SmolVLA: approach-aware episode − chunk conditioning (pp) | +2.92 | svla_cfgph\|cond, svla_cfg_ph_row\|cond |
| X-VLA: approach-aware episode − chunk conditioning (pp) | +2.00 | xvla_cfgph\|cond, xvla_cfg_ph_row\|cond |
| π0.5: object-centric episode − chunk conditioning (pp) | −3.92 | cfg_v2_ep\|cond, cfg_v2_row\|cond |
| X-VLA: object-centric chunk − episode filtering (pp) | −4.67 | xvla_drop_v2_row\|bare, xvla_drop_v2_ep\|bare |
| X-VLA: random chunk − random episode selection (pp) | +3.58 | xvla_rand_row\|bare, xvla_rand_ep\|bare |
| object-centric weighting: episode − chunk, per VLA | 3 positive, 0 equal, 0 negative | π0.5 +4.33; SmolVLA +0.50; X-VLA +1.92 |
| approach-aware weighting: episode − chunk, per VLA | 1 positive, 0 equal, 2 negative | π0.5 −3.42; SmolVLA +1.25; X-VLA −2.25 |
| approach-aware chunk weighting − plain BC, per VLA (pp) | 3 positive, 0 equal, 0 negative | π0.5 +0.67; SmolVLA +0.50; X-VLA +2.25 |

## V.D Retaining quality tags during fine-tuning

**Fig. 7. Conditioning stage and inference context, approach-aware episode labels, relative to plain BC / plain FT / untagged inference**

| VLA | group | bar | SR (%) | relative (%) | SE (%) | note | cells |
|---|---|---|---|---|---|---|---|
| π0.5 | BC pretraining / plain FT | untagged inference | 57.42 | 0.0 | 1.9 | the plain-BC reference band; native protocol | bc\|bare |
| π0.5 | BC pretraining / plain FT | tag-5 inference | 58.33 | +1.6 | 1.9 | Sec. V.D only; not a Fig. 7 bar | bc\|cond |
| π0.5 | BC pretraining / tagged FT | untagged inference | 57.75 | +0.6 | 1.8 |  | bct\|bare |
| π0.5 | BC pretraining / tagged FT | tag-5 inference | 57.08 | −0.6 | 1.9 | native protocol | bct\|cond |
| π0.5 | conditioning / plain FT | untagged inference | 57.00 | −0.7 | 1.9 | native protocol | cfgnp\|bare |
| π0.5 | conditioning / plain FT | tag-5 inference | 57.75 | +0.6 | 1.9 |  | cfgnp\|cond |
| π0.5 | conditioning / tagged FT | untagged inference | 57.83 | +0.7 | 1.8 |  | cfg_ph_ep\|bare |
| π0.5 | conditioning / tagged FT | tag-5 inference | 58.00 | +1.0 | 1.9 | native protocol | cfg_ph_ep\|cond |
| SmolVLA | BC pretraining / plain FT | untagged inference | 5.08 | 0.0 | 11.7 | the plain-BC reference band; native protocol | svla_bc\|bare |
| SmolVLA | BC pretraining / plain FT | tag-5 inference | 4.42 | −13.1 | 11.0 | Sec. V.D only; not a Fig. 7 bar | svla_bc\|cond |
| SmolVLA | BC pretraining / tagged FT | untagged inference | 5.42 | +6.6 | 11.9 |  | svla_bct\|bare |
| SmolVLA | BC pretraining / tagged FT | tag-5 inference | 5.92 | +16.4 | 12.5 | native protocol | svla_bct\|cond |
| SmolVLA | conditioning / plain FT | untagged inference | 6.25 | +23.0 | 12.3 | native protocol | svla_cfgnp\|bare |
| SmolVLA | conditioning / plain FT | tag-5 inference | 3.83 | −24.6 | 10.4 |  | svla_cfgnp\|cond |
| SmolVLA | conditioning / tagged FT | untagged inference | 6.92 | +36.1 | 13.0 |  | svla_cfgph\|bare |
| SmolVLA | conditioning / tagged FT | tag-5 inference | 8.83 | +73.8 | 14.7 | native protocol | svla_cfgph\|cond |
| X-VLA | BC pretraining / plain FT | untagged inference | 25.75 | 0.0 | 3.9 | the plain-BC reference band; native protocol | xvla_bc\|bare |
| X-VLA | BC pretraining / tagged FT | untagged inference | 25.58 | −0.6 | 4.0 |  | xvla_bct\|bare |
| X-VLA | BC pretraining / tagged FT | tag-5 inference | 27.50 | +6.8 | 4.0 | native protocol | xvla_bct\|cond |
| X-VLA | conditioning / plain FT | untagged inference | 24.33 | −5.5 | 4.0 | native protocol | xvla_cfgnp\|bare |
| X-VLA | conditioning / plain FT | tag-5 inference | 25.67 | −0.3 | 4.0 |  | xvla_cfgnp\|cond |
| X-VLA | conditioning / tagged FT | untagged inference | 28.33 | +10.0 | 4.1 |  | xvla_cfgph\|bare |
| X-VLA | conditioning / tagged FT | tag-5 inference | 27.83 | +8.1 | 4.1 | native protocol | xvla_cfgph\|cond |

**In-text numbers**

| claim | value | cells / per comparison |
|---|---|---|
| π0.5: tagged − plain FT after conditioning, tag-5 inference (pp) | +0.25 | cfg_ph_ep\|cond, cfgnp\|cond |
| π0.5: tagged − plain FT after conditioning, untagged inference (pp) | +0.83 | cfg_ph_ep\|bare, cfgnp\|bare |
| SmolVLA: tagged − plain FT after conditioning, tag-5 inference (pp) | +5.00 | svla_cfgph\|cond, svla_cfgnp\|cond |
| SmolVLA: tagged − plain FT after conditioning, untagged inference (pp) | +0.67 | svla_cfgph\|bare, svla_cfgnp\|bare |
| X-VLA: tagged − plain FT after conditioning, tag-5 inference (pp) | +2.17 | xvla_cfgph\|cond, xvla_cfgnp\|cond |
| X-VLA: tagged − plain FT after conditioning, untagged inference (pp) | +4.00 | xvla_cfgph\|bare, xvla_cfgnp\|bare |
| π0.5: adding tag 5 at inference after conditioning + plain FT (pp) | +0.75 | cfgnp\|cond, cfgnp\|bare |
| π0.5: adding tag 5 at inference after conditioning + tagged FT (pp) | +0.17 | cfg_ph_ep\|cond, cfg_ph_ep\|bare |
| SmolVLA: adding tag 5 at inference after conditioning + plain FT (pp) | −2.42 | svla_cfgnp\|cond, svla_cfgnp\|bare |
| SmolVLA: adding tag 5 at inference after conditioning + tagged FT (pp) | +1.92 | svla_cfgph\|cond, svla_cfgph\|bare |
| X-VLA: adding tag 5 at inference after conditioning + plain FT (pp) | +1.33 | xvla_cfgnp\|cond, xvla_cfgnp\|bare |
| X-VLA: adding tag 5 at inference after conditioning + tagged FT (pp) | −0.50 | xvla_cfgph\|cond, xvla_cfgph\|bare |
| π0.5: after plain-BC pretraining, tagged FT + tag-5 inference − plain FT + untagged inference (pp) | −0.33 | bct\|cond, bc\|bare |
| SmolVLA: after plain-BC pretraining, tagged FT + tag-5 inference − plain FT + untagged inference (pp) | +0.83 | svla_bct\|cond, svla_bc\|bare |
| X-VLA: after plain-BC pretraining, tagged FT + tag-5 inference − plain FT + untagged inference (pp) | +1.75 | xvla_bct\|cond, xvla_bc\|bare |
| π0.5: after plain-BC pretraining and plain FT, tag-5 − untagged inference (pp) | +0.92 | bc\|cond, bc\|bare |
| SmolVLA: after plain-BC pretraining and plain FT, tag-5 − untagged inference (pp) | −0.67 | svla_bc\|cond, svla_bc\|bare |

## V.E Scaling with dataset size

**Fig. 6. Scaling with pretraining data size, relative to each VLA's fixed 5K plain BC**

| VLA | group | bar | SR (%) | relative (%) | SE (%) | note | cells |
|---|---|---|---|---|---|---|---|
| π0.5 | plain BC | 1K | 48.42 | −15.7 | 2.0 | plain FT / untagged inference | s1000bc\|bare |
| π0.5 | plain BC | 2.5K | 56.75 | −1.2 | 1.9 | plain FT / untagged inference | s2500bc\|bare |
| π0.5 | plain BC | 5K | 57.42 | 0.0 | 1.9 | plain FT / untagged inference | bc\|bare |
| π0.5 | conditioning | 1K | 45.92 | −20.0 | 1.8 | tagged FT / tag-5 inference | s1000cfg\|cond |
| π0.5 | conditioning | 2.5K | 53.83 | −6.2 | 1.9 | tagged FT / tag-5 inference | s2500cfg\|cond |
| π0.5 | conditioning | 5K | 58.00 | +1.0 | 1.9 | tagged FT / tag-5 inference | cfg_ph_ep\|cond |
| SmolVLA | plain BC | 1K | 1.67 | −67.2 | 7.1 | plain FT / untagged inference | svla_s1000bc\|bare |
| SmolVLA | plain BC | 2.5K | 3.08 | −39.3 | 9.4 | plain FT / untagged inference | svla_s2500bc\|bare |
| SmolVLA | plain BC | 5K | 5.08 | 0.0 | 11.7 | plain FT / untagged inference | svla_bc\|bare |
| SmolVLA | conditioning | 1K | 2.33 | −54.1 | 8.3 | tagged FT / tag-5 inference | svla_s1000cfg\|cond |
| SmolVLA | conditioning | 2.5K | 5.33 | +4.9 | 11.8 | tagged FT / tag-5 inference | svla_s2500cfg\|cond |
| SmolVLA | conditioning | 5K | 8.83 | +73.8 | 14.7 | tagged FT / tag-5 inference | svla_cfgph\|cond |
| X-VLA | plain BC | 1K | 10.75 | −58.3 | 3.2 | plain FT / untagged inference | xvla_s1000bc\|bare |
| X-VLA | plain BC | 2.5K | 21.50 | −16.5 | 3.9 | plain FT / untagged inference | xvla_s2500bc\|bare |
| X-VLA | plain BC | 5K | 25.75 | 0.0 | 3.9 | plain FT / untagged inference | xvla_bc\|bare |
| X-VLA | conditioning | 1K | 13.17 | −48.9 | 3.2 | tagged FT / tag-5 inference | xvla_s1000cfg\|cond |
| X-VLA | conditioning | 2.5K | 20.75 | −19.4 | 3.8 | tagged FT / tag-5 inference | xvla_s2500cfg\|cond |
| X-VLA | conditioning | 5K | 27.83 | +8.1 | 4.1 | tagged FT / tag-5 inference | xvla_cfgph\|cond |

**In-text numbers**

| claim | value | cells / per comparison |
|---|---|---|
| π0.5: conditioning − plain BC at matched pretraining size | 1 positive, 0 equal, 2 negative | 1K −2.50; 2.5K −2.92; 5K +0.58 |
| π0.5: plain BC at 1K (%) | 48.42 | s1000bc\|bare |
| π0.5: conditioning at 1K (%) | 45.92 | s1000cfg\|cond |
| π0.5: plain BC at 2.5K (%) | 56.75 | s2500bc\|bare |
| π0.5: conditioning at 2.5K (%) | 53.83 | s2500cfg\|cond |
| π0.5: plain BC at 5K (%) | 57.42 | bc\|bare |
| π0.5: conditioning at 5K (%) | 58.00 | cfg_ph_ep\|cond |
| SmolVLA: conditioning − plain BC at matched pretraining size | 3 positive, 0 equal, 0 negative | 1K +0.67; 2.5K +2.25; 5K +3.75 |
| SmolVLA: plain BC at 1K (%) | 1.67 | svla_s1000bc\|bare |
| SmolVLA: conditioning at 1K (%) | 2.33 | svla_s1000cfg\|cond |
| SmolVLA: plain BC at 2.5K (%) | 3.08 | svla_s2500bc\|bare |
| SmolVLA: conditioning at 2.5K (%) | 5.33 | svla_s2500cfg\|cond |
| SmolVLA: plain BC at 5K (%) | 5.08 | svla_bc\|bare |
| SmolVLA: conditioning at 5K (%) | 8.83 | svla_cfgph\|cond |
| X-VLA: conditioning − plain BC at matched pretraining size | 2 positive, 0 equal, 1 negative | 1K +2.42; 2.5K −0.75; 5K +2.08 |
| X-VLA: plain BC at 1K (%) | 10.75 | xvla_s1000bc\|bare |
| X-VLA: conditioning at 1K (%) | 13.17 | xvla_s1000cfg\|cond |
| X-VLA: plain BC at 2.5K (%) | 21.50 | xvla_s2500bc\|bare |
| X-VLA: conditioning at 2.5K (%) | 20.75 | xvla_s2500cfg\|cond |
| X-VLA: plain BC at 5K (%) | 25.75 | xvla_bc\|bare |
| X-VLA: conditioning at 5K (%) | 27.83 | xvla_cfgph\|cond |

## V.F Training-seed replication

**Table I. Training-seed replication. Entries are mean ± sample SD of BC-normalized success-rate differences (%) over three seeds per VLA, using approach-aware reward**

| Comparison | π0.5 | SmolVLA | X-VLA |
|---|---:|---:|---:|
| Episode cond. − BC | +3.86 ± 5.83 | +23.42 ± 45.42 | +5.62 ± 5.39 |
| Cond.: episode − chunk | −0.90 ± 8.52 | +24.71 ± 32.64 | +11.36 ± 15.16 |
| Filter: chunk − episode | −1.90 ± 3.56 | +6.63 ± 28.34 | +5.59 ± 8.05 |
| Tagged − plain: untagged | +0.36 ± 4.30 | +9.58 ± 12.31 | +7.86 ± 6.81 |
| Tagged − plain: tag 5 | +0.93 ± 4.28 | +62.29 ± 31.29 | +8.93 ± 1.95 |

**Table I per training seed (%)**

| VLA | comparison | seed | value |
|---|---|---|---|
| π0.5 | Episode cond. − BC | 42 | +1.02 |
| π0.5 | Episode cond. − BC | 43 | 0.00 |
| π0.5 | Episode cond. − BC | 44 | +10.56 |
| SmolVLA | Episode cond. − BC | 0 | +73.77 |
| SmolVLA | Episode cond. − BC | 1 | +10.96 |
| SmolVLA | Episode cond. − BC | 2 | −14.47 |
| X-VLA | Episode cond. − BC | 0 | +8.09 |
| X-VLA | Episode cond. − BC | 1 | +9.33 |
| X-VLA | Episode cond. − BC | 2 | −0.56 |
| π0.5 | Cond.: episode − chunk | 42 | +6.10 |
| π0.5 | Cond.: episode − chunk | 43 | −10.39 |
| π0.5 | Cond.: episode − chunk | 44 | +1.60 |
| SmolVLA | Cond.: episode − chunk | 0 | +57.38 |
| SmolVLA | Cond.: episode − chunk | 1 | +24.66 |
| SmolVLA | Cond.: episode − chunk | 2 | −7.89 |
| X-VLA | Cond.: episode − chunk | 0 | +7.77 |
| X-VLA | Cond.: episode − chunk | 1 | +27.99 |
| X-VLA | Cond.: episode − chunk | 2 | −1.68 |
| π0.5 | Filter: chunk − episode | 42 | 0.00 |
| π0.5 | Filter: chunk − episode | 43 | −6.01 |
| π0.5 | Filter: chunk − episode | 44 | +0.32 |
| SmolVLA | Filter: chunk − episode | 0 | +37.70 |
| SmolVLA | Filter: chunk − episode | 1 | −17.81 |
| SmolVLA | Filter: chunk − episode | 2 | 0.00 |
| X-VLA | Filter: chunk − episode | 0 | +1.62 |
| X-VLA | Filter: chunk − episode | 1 | +0.29 |
| X-VLA | Filter: chunk − episode | 2 | +14.85 |
| π0.5 | Tagged − plain: untagged | 42 | +1.45 |
| π0.5 | Tagged − plain: untagged | 43 | −4.38 |
| π0.5 | Tagged − plain: untagged | 44 | +4.00 |
| SmolVLA | Tagged − plain: untagged | 0 | +13.11 |
| SmolVLA | Tagged − plain: untagged | 1 | −4.11 |
| SmolVLA | Tagged − plain: untagged | 2 | +19.74 |
| X-VLA | Tagged − plain: untagged | 0 | +15.53 |
| X-VLA | Tagged − plain: untagged | 1 | +5.54 |
| X-VLA | Tagged − plain: untagged | 2 | +2.52 |
| π0.5 | Tagged − plain: tag 5 | 42 | +0.44 |
| π0.5 | Tagged − plain: tag 5 | 43 | −3.08 |
| π0.5 | Tagged − plain: tag 5 | 44 | +5.44 |
| SmolVLA | Tagged − plain: tag 5 | 0 | +98.36 |
| SmolVLA | Tagged − plain: tag 5 | 1 | +42.47 |
| SmolVLA | Tagged − plain: tag 5 | 2 | +46.05 |
| X-VLA | Tagged − plain: tag 5 | 0 | +8.41 |
| X-VLA | Tagged − plain: tag 5 | 1 | +11.08 |
| X-VLA | Tagged − plain: tag 5 | 2 | +7.28 |

A training seed is a replicate of the pretraining run, keyed by its pretraining seed; each cell's fine-tuning seed is recorded in cells.json. Cells whose fine-tuning seed differs from the replicate's seed:

| VLA | seed | role | cell | pretraining seed | fine-tuning seed |
|---|---|---|---|---|---|
| π0.5 | 43 | episode conditioning (tagged FT, tag-5 inference) | cfgpt43\|cond | 43 | 42 |
| π0.5 | 43 | chunk conditioning (tagged FT, tag-5 inference) | cfgrow43\|cond | 43 | 42 |
| π0.5 | 43 | episode filtering | dropep43\|bare | 43 | 42 |
| π0.5 | 43 | chunk filtering | droprow43\|bare | 43 | 42 |
| π0.5 | 43 | episode conditioning, tagged FT, untagged inference | cfgpt43\|bare | 43 | 42 |
| π0.5 | 43 | episode conditioning, plain FT, untagged inference | cfgnp43\|bare | 43 | 42 |
| π0.5 | 43 | episode conditioning, plain FT, tag-5 inference | cfgnp43\|cond | 43 | 42 |
| π0.5 | 44 | plain BC (plain FT, untagged inference) | bc44\|bare | 44 | 42 |
| π0.5 | 44 | episode conditioning (tagged FT, tag-5 inference) | cfgpt44\|cond | 44 | 42 |
| π0.5 | 44 | chunk conditioning (tagged FT, tag-5 inference) | cfgrow44\|cond | 44 | 42 |
| π0.5 | 44 | episode filtering | dropep44\|bare | 44 | 42 |
| π0.5 | 44 | chunk filtering | droprow44\|bare | 44 | 42 |
| π0.5 | 44 | episode conditioning, tagged FT, untagged inference | cfgpt44\|bare | 44 | 42 |
| π0.5 | 44 | episode conditioning, plain FT, untagged inference | cfgnp44\|bare | 44 | 42 |
| π0.5 | 44 | episode conditioning, plain FT, tag-5 inference | cfgnp44\|cond | 44 | 42 |

**In-text numbers**

| claim | value | cells / per comparison |
|---|---|---|
| seed-level episode conditioning − BC comparisons (nine VLA-seed combinations) | 6 positive, 1 equal, 2 negative |  |
| conditioning, episode − chunk: equal average over the nine VLA-seed combinations (%) | +11.72 |  |
| filtering, chunk − episode: equal average over the nine VLA-seed combinations (%) | +3.44 |  |
| per-VLA means of the Table I comparison Cond.: episode − chunk | 2 positive, 0 equal, 1 negative |  |
| per-VLA means of the Table I comparison Filter: chunk − episode | 2 positive, 0 equal, 1 negative |  |
| VLA-inference comparisons of tagged − plain FT (means over seeds) | 6 positive, 0 equal, 0 negative |  |
| seed-level tagged − plain FT differences | 15 positive, 0 equal, 3 negative |  |
| SmolVLA: tag-5 − untagged inference after conditioning + tagged FT, per training seed | 1 positive, 0 equal, 2 negative | seed 0 +1.92; seed 1 −1.42; seed 2 −2.17 |
| X-VLA: tag-5 − untagged inference after conditioning + tagged FT, per training seed | 0 positive, 0 equal, 3 negative | seed 0 −0.50; seed 1 −0.75; seed 2 −1.00 |
