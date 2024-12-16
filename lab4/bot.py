import json
from typing import Callable

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import dotenv_values

from common.start_logging import start_logging
from lab2.manual_rag import create_query_function_rag_manual
from lab3.framework_rag import create_query_function


class Bot:
    _query_function: Callable[[str], str]

    def __init__(self, query: Callable[[str], str]):
        self._query_function = query

    def start_bot(self):
        config = dotenv_values(".env")

        bot_token = config.get("TG_BOT_TOKEN")

        app = ApplicationBuilder().token(bot_token).build()

        app.add_handler(CommandHandler("start", self._handle_start_message))

        app.add_handler(CommandHandler("query", self._handle_query))

        app.add_handler(CommandHandler("help", self._handle_help))

        app.run_polling()

    async def _handle_start_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            f'Hello {update.effective_user.first_name}')


    async def _handle_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.message.text.removeprefix("/query").strip()

        if not query:
            await update.message.reply_text("Please enter a query.")

        message = await update.message.reply_text("Querying...")

        response = self._query_function(query)

        await message.edit_text(response)

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            'Reply with /query and some question'
        )

def main() -> None:
    start_logging()

    with open('../lab2/data/cards.json', 'r') as f:
        dataset = json.load(f)

    query_function = create_query_function(dataset)

    bot = Bot(query_function)

    bot.start_bot()


if __name__ == '__main__':
    main()
