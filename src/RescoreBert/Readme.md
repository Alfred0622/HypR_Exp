# CLM, MLM, RescoreBERT and pBERT

## Train CLM
```console
$ python ./train_LM.py CLM
```

If you don't want to activate wandb, then run
```console
$ WANDB_MODE=disabled python ./train_LM.py CLM
```
instead

## Predict CLM
```console
$ python ./predict_CLM.py <checkpoint-path>
```

---

## Train MLM
```console
$ python ./train_LM.py MLM
```

If you don't want to activate wandb, then run
```console
$ WANDB_MODE=disabled python ./train_LM.py MLM
```
instead

## Predict MLM
```console
$ python ./predict_MLM.py <checkpoint-path>
```

---
## Train RescoreBERT
**If you want to train RescoreBERT, you must have the score from other Language model(e.g. CLM or MLM) first**

```console
$ python ./train_RescoreBert.py <MD, MWER or MWED>
```
## Predict RescoreBERT
```console
$ python ./predict_RescoreBert.py <checkpoint-path>
```
