# DS-Bot - Telegram FAQ Bot about Data Science

An intent-classification bot (simple bag-of-words + feedforward neural
network) that answers English-language questions about Data Science by
picking a matching response from `intents.json`.

## Project Structure

```
intents.json       - intents (topics: DS profession, libraries, learning path, jobs, Kaggle...)
model.py           - neural network architecture (fully connected MLP, 2 hidden layers)
nltk_utils.py      - tokenization and stemming
train.py           - model training, saves data.pth, logs training progress
bot.py             - the Telegram bot itself, logs user questions/answers
requirements.txt   - dependencies
.env.example       - example file with the bot token
```

## Why This Architecture

The input is a bag-of-words vector of fixed length (word order doesn't
matter), and the dataset is small (dozens of examples per class). Under
these conditions, a simple fully connected network (Linear → ReLU →
Dropout, twice, followed by an output layer) trains in seconds on CPU and
overfits less than heavier architectures (LSTM/Transformer), which would
be overkill here and require significantly more data.

## Language

**The bot only handles English.** `intents.json` contains patterns and
responses in English, and `nltk_utils.py` uses the English `PorterStemmer`
together with a regex tokenizer `[a-z0-9]+` (Latin letters and digits, no
Cyrillic support). If a user writes in another language, `bot.py` detects
this via `langdetect` and replies with `NOT_ENGLISH_RESPONSE` ("Sorry, I
only understand English..."), except in two cases:
- the message is shorter than 3 characters (`MIN_LEN_FOR_LANG_CHECK`) -
  language detection on such short strings is unreliable, so they go
  straight to the model instead;
- `langdetect` fails to detect a language (e.g. emoji/numbers only) - the
  user isn't blocked in that case.

## Installation

```bash
pip install -r requirements.txt
```

You don't need to run `nltk.download('punkt')` - tokenization is done via
a custom regex, and `PorterStemmer` works algorithmically without
requiring extra downloaded data.


## Training the Model

Run from the project's root folder (the one containing `intents.json`),
since the script uses relative paths:

```bash
python train.py
```

The script:
- reads `intents.json`, builds the vocabulary and training set;
- trains the network for 1000 epochs (batch size 8, learning rate 0.001,
  hidden_size 8);
- **logs training progress** every 50 epochs to the console and to
  `training.log` (loss, hyperparameters, CPU/GPU device, vocabulary size,
  etc.);
- saves the trained model to `data.pth`.

## Running the Bot

```bash
python bot.py
```

On every user message, the bot:
- checks that the message is in English (otherwise replies with a polite
  decline);
- predicts the intent (tag) and the model's confidence;
- if confidence is above the threshold `CONF_THRESHOLD` (0.75) - replies
  with a random response from the matching intent, otherwise asks the
  user to rephrase or check `/help`;
- **logs the question and answer** to `qa_history.log` (timestamp,
  user_id, username, question text, predicted tag, confidence, answer
  text);
- writes a technical operation log (startup, errors) to `bot.log`.

Example line from `qa_history.log`:

```
2026-07-16 12:03:41,221 | user_id=123456 username='someone' | Q: 'what is data science?' | tag=what_is_ds confidence=0.94 | A: 'Data Science sits at the intersection...'
```

## Bot Commands

- `/start` - greeting and a short description of the bot;
- `/help` - example questions you can ask.

## Extending the Bot

- Add new intents (`tag`, `patterns`, `responses`) to `intents.json` and
  retrain the model with `train.py` - no changes to the bot code needed.
- The confidence threshold `CONF_THRESHOLD` in `bot.py` can be lowered or
  raised depending on how often the bot should admit it "didn't
  understand".
- To support a language other than English, you'll need to change several
  things together: translate/add intents in `intents.json`, replace the
  stemmer and (if needed) the regex tokenizer in `nltk_utils.py` with
  language-specific ones, and update the `is_english` check /
  `NOT_ENGLISH_RESPONSE` in `bot.py`.

## Notes / Known Gotchas

- **Working directory matters.** Both `train.py` and `bot.py` load
  `intents.json` (and `bot.py` also loads `data.pth`) using relative
  paths, so they must be run from the folder that actually contains those
  files.
- **The bot rejects non-English messages outright**, except for very
  short ones (< 3 characters) or ones where `langdetect` fails to detect a
  language - those are always passed through to the model.
