# GRU model (`models/gru.py`)

A sequence-based recommender in the style of **GRU4Rec** (Hidasi et al., 2016). It reads a user's
items **in time order** with a GRU (a type of recurrent neural network) and predicts the next item.

## Why it's in the project

Each of our four models represents users in a different way:

| model | idea | how it represents a user |
|---|---|---|
| `mostpop` | popularity | doesn't (everyone gets the same list) |
| `itemcf` | similar items | the set of items they have |
| `mf` | learned vectors | one learned vector per user ID |
| **`gru`** | **order matters** | **built from their recent items, in time order** |

The GRU is the only model that uses the **order** of interactions. The question for our project is
whether paying attention to recent behaviour changes how much a model favours popular items.

## How it works

1. **Item vectors.** Every item gets a learned vector of 64 numbers. Unlike MF, users have no stored
   vector.
2. **Reading the history.** We take a user's 10 most recent training items, oldest first, and feed
   their vectors through the GRU one at a time. After each item, the GRU outputs a 64-number
   "state" that summarizes what the user has bought so far.
3. **Score.** score(user, item) = GRU's final state · item vector (dot product). It's the same
   operation MF uses, so scoring every item for every user is just as fast.
4. **Training (next-item prediction).** For a history `A → B → C`, the GRU learns to:
   - predict `B` after reading `A`
   - predict `C` after reading `A, B`

   At each step we use **BPR loss**: push the real next item's score above the score of a random item.
5. **Recommending.** `run.py` takes the top 10 scores, skipping items the user already has, and
   computes the same metrics as the other models: Recall, NDCG, head/tail recall, ARP, Gini and
   Coverage.

Users with only 1 training item have nothing to predict, so they are skipped during training. They
still get recommendations, which come from their one item.

## How to run

```
python split.py Clothing_Shoes_and_Jewelry 2
python run.py Clothing_Shoes_and_Jewelry 2 gru     # appends a row to results.csv
```

It prints the loss after each epoch; the loss should go down. If your machine is too small for the
full dataset, run it on Google Colab.

## Settings (top of `models/gru.py`)

| setting | default | meaning |
|---|---|---|
| `DIM` | 64 | size of item vectors and GRU state |
| `EPOCHS` | 5 | passes over the training users |
| `LR` | 0.001 | Adam learning rate |
| `MAX_LEN` | 10 | how many recent items the GRU reads per user |
| `TRAIN_BATCH` | 1024 | users per training step |

## Things to know

- **It needs time order.** It assumes the training rows of each user are sorted by time. `split.py`
  already writes them that way, so don't reshuffle the train file.
- **Short histories.** Our users have only about 1–3 interactions each, so the GRU has little order to
  learn from. Don't be surprised if it doesn't beat MF or item-CF on accuracy. A sequence model
  that doesn't help on short histories is a legitimate finding.
- **Random negatives.** Negatives are sampled uniformly from all items, the same as in `mf`. Sampling
  them in proportion to popularity is a possible popularity-bias experiment.
- **Memory.** There is no per-user vector table, so it needs less GPU memory than `mf`. The
  histories take about 2 GB of GPU memory for 22M users.

## Sanity check (synthetic toy data, not real results)

On a made-up dataset of 5,000 users, we trained for 200 epochs (raised just for this test). The
loss fell from 1.43 to 0.12, and Recall@10 was 0.059 (MostPop: 0.006, ItemCF: 0.10). Real numbers
will come from the Clothing_Shoes_and_Jewelry run.
