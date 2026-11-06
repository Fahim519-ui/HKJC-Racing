# Final-scoring DRY RUN on h_s2 (not the test)

Predictions: `combined_predictions.parquet`, SHA-256 8bd04281122d5ba681af07c29ff3f8d0901157612cb88ae636b18b4ecee2f8b1.
h_s2 is the Stage-2 fitting window: p_joint and p_market_refit were fitted on it, so these numbers are in-sample. This checks the pipeline only; it is not evidence.

## Verdicts

Races scored: 423  (bootstrap: race-day clusters, 43 race days)
- `p_uniform`: log loss 2.5173, pseudo-R2 0.0000; vs refit market: worse by -0.4799 (95% CI -0.5766 to -0.3826); vs raw market: worse by -0.4789 (95% CI -0.5716 to -0.3861)
- `p_market`: log loss 2.0384, pseudo-R2 0.1903; vs refit market: no clear difference (-0.0010, 95% CI -0.0050 to +0.0032); vs raw market: no comparison
- `p_power`: log loss 2.0389, pseudo-R2 0.1901; vs refit market: no clear difference (-0.0015, 95% CI -0.0067 to +0.0036); vs raw market: no clear difference (-0.0005, 95% CI -0.0100 to +0.0085)
- `p_shin`: log loss 2.0405, pseudo-R2 0.1894; vs refit market: no clear difference (-0.0030, 95% CI -0.0121 to +0.0053); vs raw market: no clear difference (-0.0021, 95% CI -0.0152 to +0.0103)
- `p_market_refit`: log loss 2.0374, pseudo-R2 0.1906; vs refit market: no comparison; vs raw market: no clear difference (+0.0010, 95% CI -0.0032 to +0.0050)
- `p_cl`: log loss 2.3154, pseudo-R2 0.0802; vs refit market: worse by -0.2779 (95% CI -0.3452 to -0.2116); vs raw market: worse by -0.2770 (95% CI -0.3418 to -0.2129)
- `p_lgbm`: log loss 2.3171, pseudo-R2 0.0795; vs refit market: worse by -0.2797 (95% CI -0.3578 to -0.2041); vs raw market: worse by -0.2787 (95% CI -0.3542 to -0.2062)
- `p_joint`: log loss 2.0222, pseudo-R2 0.1967; vs refit market: no clear difference (+0.0152, 95% CI -0.0007 to +0.0325); vs raw market: better by +0.0162 (95% CI +0.0003 to +0.0335)

## Scoreboard

```
         model  races  log_loss  pseudo_r2  brier   top1  vs_market_gain  vs_market_lo  vs_market_hi vs_market_beats  vs_refit_gain  vs_refit_lo  vs_refit_hi vs_refit_beats
     p_uniform    423    2.5173     0.0000 0.9244 0.0821         -0.4789       -0.5716       -0.3861           False        -0.4799      -0.5766      -0.3826          False
      p_market    423    2.0384     0.1903 0.8258 0.3026             NaN           NaN           NaN             NaN        -0.0010      -0.0050       0.0032          False
       p_power    423    2.0389     0.1901 0.8271 0.3026         -0.0005       -0.0100        0.0085           False        -0.0015      -0.0067       0.0036          False
        p_shin    423    2.0405     0.1894 0.8264 0.3026         -0.0021       -0.0152        0.0103           False        -0.0030      -0.0121       0.0053          False
p_market_refit    423    2.0374     0.1906 0.8259 0.3026          0.0010       -0.0032        0.0050           False            NaN          NaN          NaN            NaN
          p_cl    423    2.3154     0.0802 0.8920 0.2104         -0.2770       -0.3418       -0.2129           False        -0.2779      -0.3452      -0.2116          False
        p_lgbm    423    2.3171     0.0795 0.8899 0.2151         -0.2787       -0.3542       -0.2062           False        -0.2797      -0.3578      -0.2041          False
       p_joint    423    2.0222     0.1967 0.8202 0.3073          0.0162        0.0003        0.0335            True         0.0152      -0.0007       0.0325          False
```

Gain > 0 means the model has lower log loss than the baseline. 'Better' needs the whole 95% CI above 0.
