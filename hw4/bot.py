import json
import logging
import os
import random
import sqlite3
from datetime import datetime, timedelta, timezone

import torch
from dotenv import load_dotenv
from langdetect import DetectorFactory, LangDetectException, detect_langs
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Make langdetect deterministic - by default it uses a PRNG internally and
# can return a different answer for the same text across calls, which is
# especially visible on short inputs.
DetectorFactory.seed = 0

# ---------------------------------------------------------------------------
# Logging
# 1) general technical bot log (startup, errors) -> bot.log
# 2) separate log of user questions/answers -> qa_history.log
# ---------------------------------------------------------------------------
logger = logging.getLogger("bot")
logger.setLevel(logging.INFO)
_bot_fh = logging.FileHandler("bot.log", encoding="utf-8")
_bot_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_bot_fh)
logger.addHandler(logging.StreamHandler())

# ---------------------------------------------------------------------------
# Q/A history storage: SQLite instead of a plain-text log file.
# Keeps the same information (user, question, tag, confidence, answer) but
# lets us query/analyze it and, importantly, prune old rows so it doesn't
# grow forever.
# ---------------------------------------------------------------------------
QA_DB_PATH = "qa_history.db"
QA_RETENTION_DAYS = 90  # how long to keep individual Q/A rows

qa_conn = sqlite3.connect(QA_DB_PATH, check_same_thread=False)
qa_conn.execute(
    """
    CREATE TABLE IF NOT EXISTS qa_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        user_id INTEGER,
        username TEXT,
        question TEXT,
        tag TEXT,
        confidence REAL,
        answer TEXT
    )
    """
)
qa_conn.execute(
    "CREATE INDEX IF NOT EXISTS idx_qa_log_ts ON qa_log (ts)"
)
qa_conn.commit()


def log_qa(user_id, username, question, tag, confidence, answer):
    """Persists one Q/A exchange to the database."""
    qa_conn.execute(
        "INSERT INTO qa_log (ts, user_id, username, question, tag, confidence, answer) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            user_id,
            username,
            question,
            tag,
            confidence,
            answer,
        ),
    )
    qa_conn.commit()


def cleanup_old_qa(days: int = QA_RETENTION_DAYS):
    """Deletes Q/A rows older than `days`. Returns number of rows removed."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    cur = qa_conn.execute("DELETE FROM qa_log WHERE ts < ?", (cutoff,))
    qa_conn.commit()
    return cur.rowcount


async def cleanup_old_qa_job(context: ContextTypes.DEFAULT_TYPE):
    """JobQueue callback: runs cleanup_old_qa daily and logs the result."""
    removed = cleanup_old_qa()
    if removed:
        logger.info(f"QA history cleanup: removed {removed} rows older than {QA_RETENTION_DAYS} days")

# ---------------------------------------------------------------------------
# Load model and data
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

data = torch.load("data.pth", map_location=device)

model = NeuralNet(
    data["input_size"], data["hidden_size"], data["output_size"]
).to(device)
model.load_state_dict(data["model_state"])
model.eval()

all_words = data["all_words"]
tags = data["tags"]

BOT_NAME = "DS-Bot"
CONF_THRESHOLD = 0.75
FALLBACK_RESPONSE = (
    "Sorry, I didn't understand that. Try rephrasing your question, "
    "or send /help to see what I can do."
)
NOT_ENGLISH_RESPONSE = (
    "Sorry, I only understand English at the moment. "
    "Please ask your question in English."
)

# Minimum message length (in characters) below which language detection is
# unreliable; short messages are passed straight to the model instead of
# being blocked. langdetect's statistical model needs a reasonable amount
# of text to work with - single short words/greetings ("hello", "thanks",
# "greeting") are frequently misclassified as other languages, so the
# threshold is set well above typical short greetings.
MIN_LEN_FOR_LANG_CHECK = 20

# Minimum confidence langdetect must report for "en" before we trust a
# non-English verdict enough to block the user.
EN_PROB_THRESHOLD = 0.7


def is_english(text: str) -> bool:
    """
    Returns True if the message is likely English (or too short/ambiguous
    to reliably tell), False only if another language is detected with
    reasonably high confidence.
    """
    stripped = text.strip()
    if len(stripped) < MIN_LEN_FOR_LANG_CHECK:
        return True
    try:
        candidates = detect_langs(stripped)
    except LangDetectException:
        # Detection failed (e.g. text with no linguistic content, like
        # emojis/numbers only) - don't block the user in that case.
        return True

    for candidate in candidates:
        if candidate.lang == "en":
            return True
        if candidate.prob >= EN_PROB_THRESHOLD:
            # Confidently some other language - block.
            return False

    # No language met the confidence bar - don't block on an unclear signal.
    return True


def get_response(message: str):
    """Runs the message through the model and returns (response, tag, confidence)."""
    tokens = tokenize(message)
    if not tokens:
        return FALLBACK_RESPONSE, "empty", 0.0

    X = bag_of_words(tokens, all_words)

    if not X.any():
        # None of the words in the message are in the trained vocabulary
        # (e.g. pure numbers, random gibberish, or words the model never
        # saw). The bag-of-words vector is then all zeros, and the model's
        # output in that case is driven purely by layer biases - it can
        # still report high confidence for some tag, but that confidence
        # is meaningless since no real signal was seen. Skip the model
        # entirely and fall back.
        return FALLBACK_RESPONSE, "unrecognized", 0.0

    X = torch.from_numpy(X).reshape(1, -1).to(device)

    with torch.no_grad():
        output = model(X)
        probs = torch.softmax(output, dim=1)
        prob, predicted = torch.max(probs, dim=1)
        tag = tags[predicted.item()]
        confidence = prob.item()

    if confidence > CONF_THRESHOLD:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"]), tag, confidence

    return FALLBACK_RESPONSE, tag, confidence


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Hi! I'm {BOT_NAME} - I answer questions about Data Science: "
        "the profession, tools, learning, and job hunting. Ask away! "
        "(English only, please 🇬🇧)"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "You can ask me things like:\n"
        "- What is Data Science?\n"
        "- What libraries do I need for DS?\n"
        "- Where should I start learning?\n"
        "- How do I find a job with no experience?\n"
        "- What is Kaggle?\n\n"
        "Please ask in English."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = update.message.text

    try:
        if not is_english(text):
            response = NOT_ENGLISH_RESPONSE
            tag, confidence = "non_english", 0.0
        else:
            response, tag, confidence = get_response(text)
    except Exception:
        logger.exception("Error while processing message")
        response = "Oops, something went wrong. Please try again."
        tag, confidence = "error", 0.0

    # Log every question/answer separately from the technical log
    log_qa(
        user_id=user.id,
        username=user.username,
        question=text,
        tag=tag,
        confidence=confidence,
        answer=response,
    )
    logger.info(f"user_id={user.id}: tag={tag}, confidence={confidence:.2f}")

    await update.message.reply_text(response)


def main():
    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN not found. Create a .env file and add the line "
            "BOT_TOKEN=your_botfather_token"
        )

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run QA history cleanup once a day (first run 60s after startup, then
    # every 24h) so qa_history.db doesn't grow forever.
    app.job_queue.run_repeating(cleanup_old_qa_job, interval=timedelta(days=1), first=60)

    logger.info("Bot started and listening for messages")
    app.run_polling()


if __name__ == "__main__":
    main()