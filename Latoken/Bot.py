import openai
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity #scikit-learn (Вычисление косинусного сходства для поиска релевантной информации)
import logging
from datetime import datetime
import random


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = openai.OpenAI(
    api_key=""
)

positive_gifs = [
    "C:/Users/Костя/PycharmProjects/Latoken/dancing-cat-dance.gif",
    "C:/Users/Костя/PycharmProjects/Latoken/engoy.gif",
    "C:/Users/Костя/PycharmProjects/Latoken/really-well-done-thomas-elms.gif",
    "C:/Users/Костя/PycharmProjects/Latoken/shreks-meme.gif",
    "C:/Users/Костя/PycharmProjects/Latoken/kitty-smiley-kitty.gif"
]

negative_gifs = [
    "C:/Users/Костя/PycharmProjects/Latoken/sad1.gif",
    "C:/Users/Костя/PycharmProjects/Latoken/sad2.gif",
    "C:/Users/Костя/PycharmProjects/Latoken/sad3.gif",
    "C:/Users/Костя/PycharmProjects/Latoken/sad4.gif",
    "C:/Users/Костя/PycharmProjects/Latoken/sad5.gif"
]


current_positive_gif_index = 0
current_negative_gif_index = 0


def initialize_database():
    conn = sqlite3.connect("rag_database.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fragments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT,
        text TEXT,
        embedding BLOB
    )
    """)
    conn.commit()
    conn.close()


def create_embeddings(texts):
    try:
        # Проверяем, что texts не пустой и содержит только строки
        if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
            logger.warning("Некорректные данные для создания эмбеддингов: пустой или невалидный текст.")
            return [np.zeros(1536) for _ in texts]  
        response = client.embeddings.create(
            model="text-embedding-ada-002",  
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Ошибка при создании эмбеддингов: {e}")
        return [np.zeros(1536) for _ in texts]  


def load_and_store_data(file_path):
    conn = sqlite3.connect("rag_database.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM fragments")
    conn.commit()
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    current_category = None
    batch = []
    for line in lines:
        if line.startswith("[") and line.endswith("]\n"):
            current_category = line.strip()[1:-1]
        elif line.strip():
            batch.append((current_category, line.strip()))
            if len(batch) >= 50:
                process_batch(cursor, batch)
                batch = []
    if batch:
        process_batch(cursor, batch)
    conn.commit()
    conn.close()

def process_batch(cursor, batch):
    categories, texts = zip(*batch)
    embeddings = create_embeddings(texts)
    cursor.executemany("""
    INSERT INTO fragments (category, text, embedding)
    VALUES (?, ?, ?)
    """, [(cat, txt, str(list(emb))) for cat, txt, emb in zip(categories, texts, embeddings)])


def retrieve_relevant_fragments(query, category=None, top_k=25):
    query_embedding = create_embeddings([query])[0]
    conn = sqlite3.connect("rag_database.db")
    cursor = conn.cursor()
    if category:
        cursor.execute("SELECT category, text, embedding FROM fragments WHERE category=?", (category,))
    else:
        cursor.execute("SELECT category, text, embedding FROM fragments")
    rows = cursor.fetchall()
    conn.close()
    results = []
    for category, text, emb in rows:
        emb_array = np.array(eval(emb))
        similarity = cosine_similarity([query_embedding], [emb_array])[0][0]
        results.append((category, text, similarity))
    categorized_results = {}
    for category, text, similarity in sorted(results, key=lambda x: x[2], reverse=True):
        if category not in categorized_results:
            categorized_results[category] = []
        if len(categorized_results[category]) < top_k:
            categorized_results[category].append((text, similarity))
    return categorized_results

def generate_response(query, context_data):
    try:
        system_prompt = (
            "Ты помощник по вопросам Latoken. Отвечай на основе предоставленного контекста."
            "Если информации недостаточно - сообщи об этом вежливо и не говори про что-то мимо Latokena"
            "Всегда предоставляй все ссылки по теме."
            "Всегда ставь эмодзи там, где это будет уместно и в тему"
            "Ссылки всегда форматируй в слова"
            "Если информации недостаточно или она никак не относится к Latoken  - Обязательно скажи 'Извините'"
        )
        context_parts = []
        for category, fragments in context_data.items():
            context_parts.append(f"=== {category} ===")
            for text, _ in fragments:
                context_parts.append(text)
        full_context = "\n".join(context_parts)
        max_context_length = 3000
        if len(full_context) > max_context_length:
            full_context = full_context[:max_context_length] + "\n[...]"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Контекст:\n{full_context}\nВопрос: {query}"}
            ],
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {e}")
        return "Произошла ошибка при обработке запроса."

def generate_test_question(last_query, context_data):
    try:

        context_parts = []
        for category, fragments in context_data.items():
            context_parts.append(f"=== {category} ===")
            for text, _ in fragments:
                context_parts.append(text)
        full_context = "\n".join(context_parts)
        system_prompt = (
            "Ты помощник по созданию тестовых вопросов. Сформулируй вопрос на основе предоставленного контекста и последнего запроса пользователя."
            "Вопрос должен быть релевантным последнему запросу и сложным, но четким."
            "Сгенерируй один правильный и два неправильных варианта ответа."
            "Вариантов ответов должно быть строго 3, не больше, ни меньше. Следи за этим."
            "Формат ответа: сначала вопрос, затем три варианта ответа, разделённые новой строкой."
            "Так же нумеровка должна полностью отсутствовать"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Последний запрос: {last_query}\nКонтекст:\n{full_context}"}
            ],
            max_tokens=150,
            temperature=0.7,
            top_p=0.9
        )
        response_lines = response.choices[0].message.content.strip().split("\n")
        if len(response_lines) < 4:
            logger.error("Недостаточно данных для формирования вопроса и ответов.")
            return None, None, None
        question = response_lines[0]
        options = response_lines[2:5]  
        correct_answer = options[0]
        random.shuffle(options)
        correct_index = options.index(correct_answer)
        return question, options, correct_index  
    except Exception as e:
        logger.error(f"Ошибка при генерации тестового вопроса: {e}")
        return None, None, None

async def toggle_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current_state = context.user_data.get("test_mode", True)
    context.user_data["test_mode"] = not current_state
    state_text = "включен" if not current_state else "выключен"
    logger.info(f"Режим тестирования {state_text}")
    await update.message.reply_text(
        f"Режим тестирования {state_text}.",
        parse_mode="Markdown"
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        ["📚 Расскажи все о Latoken", "🏆 О хакатоне"],  # Первая строка кнопок
        ["🤝 Культура Latoken", "🔄 Переключить режим тестирования"]  # Вторая строка кнопок
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "👋 *Привет!* Я — твой помощник по вопросам Latoken. 😊\n"
        "Нажми одну из кнопок или задай свой вопрос в текстовом виде.\n"
        "*Ты сейчас в режиме 'тестирования'.*\n"
        "В этом режиме, после того, как задашь вопрос, будет проходить тестирование.\n"
        "Если хочешь, что бы тебя не тестировали после каждого вопроса, нажми на кнопку *'🔄 Переключить режим тестирования'*, которая расположена в меню бота.",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def send_gif(update: Update, is_correct: bool):
    global current_positive_gif_index, current_negative_gif_index

    if is_correct:
        gifs = positive_gifs
        current_index = current_positive_gif_index
        current_positive_gif_index = (current_positive_gif_index + 1) % len(positive_gifs)  # Обновляем индекс
    else:
        gifs = negative_gifs
        current_index = current_negative_gif_index
        current_negative_gif_index = (current_negative_gif_index + 1) % len(negative_gifs)  # Обновляем индекс

    gif_url = gifs[current_index]
    await update.message.reply_animation(gif_url)

def format_response(response):
    response = response.replace("[Подробнее о хакатоне]", "📖 [Подробнее о хакатоне]")
    response = response.replace("[Официальный сайт Latoken]", "🌐 [Официальный сайт Latoken]")
    response = response.replace("### ", "📌 ")  # Для основных заголовков
    response = response.replace("## ", "🔹 ")  # Для подзаголовков
    response = response.replace("# ", "🔸 ")  # Для маленьких заголовков
    response = response.replace("- Краткий ответ:", "🔍 *Краткий ответ:*")
    response = response.replace("- Подробное объяснение:", "📚 *Подробное объяснение:*")
    response = response.replace("- Цитаты из документации:", "📄 *Цитаты из документации:*")
    response = response.replace("- Ссылки на дополнительные материалы:", "🔗 *Ссылки на дополнительные материалы:*")
    return response

def load_context_from_file(file_path):

    context_data = {}
    current_category = None

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):

            current_category = line[1:-1]
            context_data[current_category] = []
        elif line and current_category:

            text = line
            embedding = create_embeddings([text])[0]
            context_data[current_category].append({"text": text, "embedding": embedding})

    return context_data

def check_if_knows_answer(query):
    try:

        query_embedding = create_embeddings([query])[0]

        conn = sqlite3.connect("rag_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM fragments")
        rows = cursor.fetchall()
        conn.close()

        similarities = []
        for row in rows:
            fragment_embedding = np.array(eval(row[0]))
            similarity = cosine_similarity([query_embedding], [fragment_embedding])[0][0]
            similarities.append(similarity)

        max_similarity = max(similarities) if similarities else 0.0

        threshold = 0.7  
        return max_similarity >= threshold
    except Exception as e:
        logger.error(f"Ошибка при проверке знания ответа: {e}")
        return False

def contains_uncertainty_phrases(response):

    uncertainty_phrases = [
        "извините",
        "прости",
        "не указано",
        "нет информации",
        "не могу сказать",
        "не знаю",
        "информация отсутствует",
        "документация не упоминает"
    ]

    response_lower = response.lower()
    for phrase in uncertainty_phrases:
        if phrase in response_lower:
            return True
    return False


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.lower()
    logger.info(f"[{datetime.now()}] Новый запрос: {user_message}")


    if user_message == "назад":
        context.user_data["testing"] = False
        keyboard = [
            ["📚 Расскажи все о Latoken", "🏆 О хакатоне"],
            ["🤝 Культура Latoken", "🔄 Переключить режим тестирования"]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        await update.message.reply_text(
            "🔙 Вы вышли с вопроса.",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
        return

    if user_message == "🔄 переключить режим тестирования":
        await toggle_test(update, context)
        return


    if context.user_data.get("testing"):
        correct_index = context.user_data["correct_index"]
        options = context.user_data["options"]  
        if user_message.isdigit():
            selected_index = int(user_message) - 1  
            if 0 <= selected_index < len(options):
                is_correct = selected_index == correct_index
                if is_correct:
                    response = "🎉 *Отлично!* Ты абсолютно прав!\n" \
                               f"Полный ответ: _{options[correct_index]}_"
                else:
                    response = "😕 *Не правильно.*\n" \
                               f"Правильный ответ: _{options[correct_index]}_"
                context.user_data["testing"] = False
                keyboard = [
                    ["📚 Расскажи все о Latoken", "🏆 О хакатоне"],
                    ["🤝 Культура Latoken", "🔄 Переключить режим тестирования"]
                ]
                reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
                await update.message.reply_text(response, reply_markup=reply_markup, parse_mode="Markdown")
                await send_gif(update, is_correct)
                return
        await update.message.reply_text("🤔 Пожалуйста, выбери номер ответа или напиши 'назад', чтобы выйти.")
        return


    if user_message == "все о latoken":
        query = "Расскажи о компании Latoken"
        category = None
    elif user_message == "все о хакатоне":
        query = "Расскажи о хакатоне"
        category = "Хакатон"
    elif user_message == "все о культуре latoken":
        query = "Расскажи о культуре Latoken"
        category = "Культура"
    else:
        query = user_message
        category = None

    expanded_query = expand_query(query)
    relevant_fragments = retrieve_relevant_fragments(expanded_query, category)
    generating_message = await update.message.reply_text("⏳ Бот генерирует... [0%]")
    try:
        for progress in range(10, 110, 10):  # От 10% до 100%
            if progress < 100:
                await context.bot.edit_message_text(
                    chat_id=update.message.chat_id,
                    message_id=generating_message.message_id,
                    text=f"⏳ Бот генерирует... [{progress}%]"
                )


        gpt_response = generate_response(expanded_query, relevant_fragments)
        formatted_response = format_response(gpt_response)


        await context.bot.delete_message(chat_id=update.message.chat_id, message_id=generating_message.message_id)

        await update.message.reply_text(formatted_response, parse_mode="Markdown")

        if context.user_data.get("test_mode", True):
            if contains_uncertainty_phrases(gpt_response):
                await update.message.reply_text(
                    "😔 Не могу проверить ваши знания, так как сам не знаю на это ответ.",
                    parse_mode="Markdown"
                )
                await send_gif(update, is_correct=False)  # Отправляем грустную гифку
                return

            test_question, options, correct_index = generate_test_question(query, relevant_fragments)
            if test_question and options:
                options_text = "\n".join([f"{i + 1}. {option}" for i, option in enumerate(options)])
                message_text = (
                    f"🧠 *Тестовый вопрос:* {test_question}\n"
                    f"📝 *Варианты ответов:*\n{options_text}\n"
                    "🎯 Выбери номер правильного ответа или напиши 'назад', чтобы выйти:"
                )
                keyboard = [[str(i + 1)] for i in range(len(options))]
                keyboard.append(["назад"])  # Добавляем кнопку "назад"
                reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
                context.user_data["testing"] = True
                context.user_data["correct_index"] = correct_index
                context.user_data["options"] = options
                await update.message.reply_text(message_text, reply_markup=reply_markup, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        await context.bot.delete_message(chat_id=update.message.chat_id, message_id=generating_message.message_id)
        await update.message.reply_text("Произошла ошибка при обработке запроса.", parse_mode="Markdown")


def expand_query(query):
    keywords_map = {
        "процесс найма": ["рекрутинг", "интервью", "тестирование"],
        "хакатон": ["расписание", "призы", "технологии", "формат", "участие"],
        "культура": ["ценности", "традиции", "командная работа"]
    }
    expanded = query
    for main_term, related in keywords_map.items():
        if main_term in query:
            expanded += " " + " ".join(related)
    return expanded

async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Вызываем тот же функционал, что и в /start
    await start(update, context)


def main():
    try:
        initialize_database()
        context_data = load_and_store_data("context2.txt")  # Загружаем данные один раз при запуске
        token = ""
        app = ApplicationBuilder().token(token).build()


        app.bot_data["context_data"] = context_data


        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("restart", restart))  # Новый обработчик
        app.add_handler(CommandHandler("toggle_test", toggle_test))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        commands = [
            ("start", "Начало работы"),
            ("restart", "Перезапустить бота"),  # Новая команда
        ]
        app.bot.set_my_commands(commands)

        logger.info("Бот успешно запущен")
        app.run_polling()
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}")


if __name__ == "__main__":

    main()

