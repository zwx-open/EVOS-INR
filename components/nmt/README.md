# NMT Algorithm 

## Setup

(TODO)
```
python setup.py
```

## Example usage

```
# import NMT wrapper class
from nmt import NMT

# ----------
# SOME CODE
# ----------

# nmt setup
nmt = NMT(model,
            train_configs.iterations,
            (T, C),
            exp_configs.scheduler_type,
            exp_configs.strategy_type,
            exp_configs.mt_ratio,
            exp_configs.top_k,
            save_samples_path=train_configs.sampling_path,
            save_losses_path=train_configs.loss_path,
            save_name=None,
            save_interval=train_configs.save_interval)

# ----------
# SOME CODE
# ----------

for step in total_steps:
    # mt sampling
    sampled_x, sampled_y, full_preds = nmt.sample(step, x, y)

    # forward pass 
    sampled_preds = model(sampled_x)
    # calculate loss
    loss = objective(sampled_preds, sampled_y)
    # backward pass
    loss.backward()
    optimizer.step()

    # ----------
    # SOME CODE
    # ----------

```
