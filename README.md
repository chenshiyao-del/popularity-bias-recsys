# Popularity bias in recommenders

Amazon Reviews 2023 data: https://amazon-reviews-2023.github.io/

```
pip install -r requirements.txt
python prepare.py Clothing_Shoes_and_Jewelry.jsonl Clothing_Shoes_and_Jewelry   # clean reviews
python split.py Clothing_Shoes_and_Jewelry 2                                    # 2-core, leave-last-out split
python run.py Clothing_Shoes_and_Jewelry 2 itemcf                               # or mostpop / mf; appends to results.csv
```
