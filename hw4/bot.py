import json
import logging
import os
import random

import torch
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
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

qa_logger = logging.getLogger("qa")
qa_logger.setLevel(logging.INFO)
_qa_fh = logging.FileHandler("qa_history.log", encoding="utf-8")
_qa_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
qa_logger.addHandler(_qa_fh)

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
# being blocked.
MIN_LEN_FOR_LANG_CHECK = 3


def is_english(text: str) -> bool:
    """
    Returns True if the message is likely English (or too short/ambiguous
    to reliably tell), False if it's confidently detected as another
    language.
    """
    stripped = text.strip()
    if len(stripped) < MIN_LEN_FOR_LANG_CHECK:
        return True
    try:
        return detect(stripped) == "en"
    except LangDetectException:
        # Detection failed (e.g. text with no linguistic content, like
        # emojis/numbers only) — don't block the user in that case.
        return True


def get_response(message: str):
    """Runs the message through the model and returns (response, tag, confidence)."""
    tokens = tokenize(message)
    if not tokens:
        return FALLBACK_RESPONSE, "empty", 0.0

    X = bag_of_words(tokens, all_words)
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
        f"Hi! I'm {BOT_NAME} — I answer questions about Data Science: "
        "the profession, tools, learning, and job hunting. Ask away! "
        "(English only, please 🇬🇧)"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "You can ask me things like:\n"
        "— What is Data Science?\n"
        "— What libraries do I need for DS?\n"
        "— Where should I start learning?\n"
        "— How do I find a job with no experience?\n"
        "— What is Kaggle?\n\n"
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
    qa_logger.info(
        f"user_id={user.id} username={user.username!r} | "
        f"Q: {text!r} | tag={tag} confidence={confidence:.2f} | A: {response!r}"
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

    logger.info("Bot started and listening for messages")
    app.run_polling()


if __name__ == "__main__":
    main()