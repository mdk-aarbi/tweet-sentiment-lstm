# Tweet Sentiment LSTM

Three-class tweet sentiment analysis (negative, neutral, positive) with LSTMs. Deep Learning course assignment, Superior University, Lahore.
A bidirectional LSTM trained from scratch (Model A) is compared with the same network using frozen GloVe Twitter 100d embeddings (Model B).
The trained model also runs in the browser: **https://tweet-sentiment-lstm.vercel.app**

## Results on the TweetEval test set (12,284 tweets)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Always guess neutral | 48.33% | n/a |
| A: BiLSTM, random embeddings | 61.01% | 0.5935 |
| B: BiLSTM, frozen GloVe Twitter | 63.73% | 0.6257 |

Single seeded run (seed 42). Confusion matrices, training curves and error examples are in `results/`.

## Data
TweetEval sentiment (`cardiffnlp/tweet_eval`, config `sentiment`): 45,615 train, 2,000 validation, 12,284 test tweets. It was not collected by the author. The full splits are available as a zip (see links below); `data/test_samples.csv` holds 100 test tweets.

## Repository layout
- `notebook/` the full Colab notebook with outputs
- `models/` trained Keras models and the vocabulary
- `results/` metrics, confusion matrices, training curves, error examples
- `index.html`, `weights.bin`, `manifest.json`, `vocab.json` the browser demo (plain JavaScript, no server)
- `llms.txt`, `results.json`, `robots.txt`, `sitemap.xml` files for search engines and AI agents
- `predict.py` command line prediction: `python predict.py "your tweet here"`
- `Assignment_Report.pdf` the assignment report

## Reproduce
Open the notebook in Google Colab and run all cells. Training takes a few minutes on a GPU runtime. Exact numbers can differ slightly between runs and hardware.

## Links
- Live demo: https://tweet-sentiment-lstm.vercel.app
- GitHub repository: https://github.com/mdk-aarbi/tweet-sentiment-lstm
- Assignment report: https://github.com/mdk-aarbi/tweet-sentiment-lstm/blob/main/Assignment_Report.pdf
- Presentation: https://claude.ai/artifact/5FMzwoKnLn8Y2QHNqinDAq
- Full dataset: https://drive.google.com/file/d/1WandK1pMuvHDLzBgZ--LzfOS7Lv_BcaL/view?usp=drivesdk

## Credits
- Barbieri et al. (2020). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. Findings of EMNLP.
- Rosenthal, Farra, Nakov (2017). SemEval-2017 Task 4: Sentiment Analysis in Twitter.
- Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation.
- LinkedIn post: https://www.linkedin.com/feed/update/urn:li:activity:7532068129993089024/
