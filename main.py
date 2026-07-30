import asyncio
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent_logger import get_log_url, log_event
from interpreter import interpret_data
from planner import planner
from retriever import retrieve_dataset

TOKEN = os.environ.get("BOT_TOKEN")

app = FastAPI()

# Mount the static directory so agent_log.jsonl is wget-able via http://localhost:3000/static/agent_log.jsonl
app.mount("/static", StaticFiles(directory="static"), name="static")

ptb = Application.builder().token(TOKEN).build()


@app.on_event("startup")
async def startup():
    await ptb.initialize()
    await ptb.start()


@app.on_event("shutdown")
async def shutdown():
    await ptb.stop()
    await ptb.shutdown()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! Send me a data query, and I'll plan, retrieve, and interpret the data for you."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    print(f"User Query: {user_query}")

    # Log incoming user query
    log_event("user_query_received", {"query": user_query})

    # 1. Planning Phase
    status_message = await update.message.reply_text("🧠 Planning search strategy...")

    try:
        planner_output = planner(user_query)
        log_event("planner_completed", planner_output)

        search_keywords = planner_output.get("search_keywords", [])

        if not search_keywords:
            await status_message.edit_text("❌ Planner failed to generate search keywords.")
            log_event("planner_error", {"reason": "No search keywords generated"})
            return

        # 2. Retrieval Phase
        await status_message.edit_text(f"🔍 Searching the web for: `{search_keywords[0]}`...", parse_mode="Markdown")

        dataset_path = await asyncio.to_thread(retrieve_dataset, planner_output)

        if not dataset_path:
            await status_message.edit_text("❌ Failed to find or download a valid dataset. Try rephrasing your request.")
            log_event("retriever_error", {"reason": "No dataset found or downloaded"})
            return

        log_event(
            "retriever_completed",
            {"dataset_filename": dataset_path.name, "dataset_path": str(dataset_path)},
        )

        # 3. Interpretation Phase
        await status_message.edit_text(
            f"⚙️ Dataset downloaded (`{dataset_path.name}`). Analyzing the data...",
            parse_mode="Markdown",
        )

        # Pass get_log_url() to the interpreter!
        current_log_url = get_log_url()
        final_answer = await asyncio.to_thread(interpret_data, user_query, dataset_path, current_log_url)

        log_event(
            "interpreter_completed",
            {"final_answer": final_answer, "log_url": current_log_url},
        )

        # 4. Final Output
        await status_message.edit_text(final_answer)

    except Exception as e:
        print(f"Error: {e}")
        log_event("pipeline_exception", {"error": str(e)})
        await status_message.edit_text("⚠️ An error occurred while processing your request.")


ptb.add_handler(CommandHandler("start", start))
ptb.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


@app.post(f"/{TOKEN}")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, ptb.bot)

    await ptb.process_update(update)

    return {"status": "ok"}