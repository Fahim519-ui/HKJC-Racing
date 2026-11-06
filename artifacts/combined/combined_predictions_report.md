# Combined predictions

Source predictions: `artifacts\lightgbm_independent_probability_v2\joint_predictions.parquet`.
Rows: 10265 ({'h_s2': 5287, 'h_test': 4978}).
Market-only refit beta (fitted on h_s2): 1.0537.

## Columns

| Column | Present | Sums to 1 per race, no NaN |
|---|---|---|
| `p_uniform` | yes | True |
| `p_market` | yes | True |
| `p_power` | yes | True |
| `p_shin` | yes | True |
| `p_market_refit` | yes | True |
| `p_cl` | yes | True |
| `p_lgbm` | yes | True |
| `p_joint` | yes | True |
| `p_b_base` | NO (Track B not built yet) | - |
| `p_b_rating` | NO (Track B not built yet) | - |
| `p_b_running` | NO (Track B not built yet) | - |
| `p_b_all` | NO (Track B not built yet) | - |

Track B columns are absent because Track B has not been built. Add them to this file once it exists. The file holds predictions for h_s2 and h_test; h_test is for Week 5 one-shot scoring only and is not scored here.
