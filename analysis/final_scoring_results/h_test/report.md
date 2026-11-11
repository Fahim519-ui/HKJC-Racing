# FINAL h_test scoring (one shot)

Run at 2026-11-11T09:06:44+00:00 UTC from git 403aac4acbc86816a49e902fe4fae2c8b15956fa.
Predictions SHA-256 a1984fdaddac7fb95090044fc16b1d1a91bbb3dde7bf60c2603c259dcf40778d (matches freeze record of 2026-11-11T09:06:43+00:00).
Freeze timestamp (UTC): 2026-11-11T09:06:43+00:00. Scored races dated after the freeze day: 0 of 400 (reported, not enforced; see docs/evaluation_protocol.md).
Nothing was refitted on h_test: p_market_refit and p_joint carry h_s2 fits.

## Verdicts

Races scored: 400  (bootstrap: race-day clusters, 40 race days)
- `p_uniform`: log loss 2.5123, pseudo-R2 0.0000; vs refit market: worse by -0.4557 (95% CI -0.5306 to -0.3808); vs raw market: worse by -0.4558 (95% CI -0.5270 to -0.3849)
- `p_market`: log loss 2.0565, pseudo-R2 0.1814; vs refit market: no clear difference (+0.0001, 95% CI -0.0039 to +0.0041); vs raw market: no comparison
- `p_power`: log loss 2.0601, pseudo-R2 0.1800; vs refit market: no clear difference (-0.0035, 95% CI -0.0095 to +0.0024); vs raw market: no clear difference (-0.0036, 95% CI -0.0134 to +0.0062)
- `p_shin`: log loss 2.0640, pseudo-R2 0.1784; vs refit market: no clear difference (-0.0074, 95% CI -0.0216 to +0.0049); vs raw market: no clear difference (-0.0075, 95% CI -0.0251 to +0.0078)
- `p_market_refit`: log loss 2.0566, pseudo-R2 0.1814; vs refit market: no comparison; vs raw market: no clear difference (-0.0001, 95% CI -0.0041 to +0.0039)
- `p_cl`: log loss 2.4053, pseudo-R2 0.0426; vs refit market: worse by -0.3487 (95% CI -0.4134 to -0.2820); vs raw market: worse by -0.3488 (95% CI -0.4100 to -0.2861)
- `p_lgbm`: log loss 2.2583, pseudo-R2 0.1011; vs refit market: worse by -0.2017 (95% CI -0.2577 to -0.1537); vs raw market: worse by -0.2018 (95% CI -0.2552 to -0.1565)
- `p_joint`: log loss 2.0593, pseudo-R2 0.1803; vs refit market: no clear difference (-0.0027, 95% CI -0.0102 to +0.0051); vs raw market: no clear difference (-0.0029, 95% CI -0.0102 to +0.0054)

## Scoreboard

```
         model  races  log_loss  pseudo_r2  brier   top1  vs_market_gain  vs_market_lo  vs_market_hi vs_market_beats  vs_refit_gain  vs_refit_lo  vs_refit_hi vs_refit_beats
     p_uniform    400    2.5123     0.0000 0.9180 0.0820         -0.4558       -0.5270       -0.3849           False        -0.4557      -0.5306      -0.3808          False
      p_market    400    2.0565     0.1814 0.8272 0.2638             NaN           NaN           NaN             NaN         0.0001      -0.0039       0.0041          False
       p_power    400    2.0601     0.1800 0.8289 0.2638         -0.0036       -0.0134        0.0062           False        -0.0035      -0.0095       0.0024          False
        p_shin    400    2.0640     0.1784 0.8282 0.2638         -0.0075       -0.0251        0.0078           False        -0.0074      -0.0216       0.0049          False
p_market_refit    400    2.0566     0.1814 0.8275 0.2638         -0.0001       -0.0041        0.0039           False            NaN          NaN          NaN            NaN
          p_cl    400    2.4053     0.0426 0.8967 0.2250         -0.3488       -0.4100       -0.2861           False        -0.3487      -0.4134      -0.2820          False
        p_lgbm    400    2.2583     0.1011 0.8694 0.2275         -0.2018       -0.2552       -0.1565           False        -0.2017      -0.2577      -0.1537          False
       p_joint    400    2.0593     0.1803 0.8293 0.2650         -0.0029       -0.0102        0.0054           False        -0.0027      -0.0102       0.0051          False
```

Gain > 0 means the model has lower log loss than the baseline. 'Better' needs the whole 95% CI above 0.
