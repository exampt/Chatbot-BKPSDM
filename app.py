from flask import Flask, request, jsonify
from google import genai
import os
from dotenv import load_dotenv
import json
import numpy as np
from collections import Counter
import time
import requests

# ====================================
# LOAD ENV
# ====================================

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)


# ====================================
# LOAD FAQ DATABASE
# ====================================

with open("faq_bkpsdm.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

faq_questions = [faq["question"] for faq in faq_data]


# ====================================
# EMBEDDING FUNCTION
# ====================================

def get_embedding(text):

    for _ in range(3):
        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )

            return response.embeddings[0].values

        except Exception as e:

            print("Embedding error:", e)
            time.sleep(1)

    raise Exception("Embedding failed after 3 attempts")


# ====================================
# EMBEDDING CACHE SYSTEM
# ====================================

CACHE_FILE = "faq_embeddings_cache.json"


def load_or_create_embeddings():

    if os.path.exists(CACHE_FILE):

        print("Loading embeddings from cache...")

        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)

        return np.array(cached)

    print("Creating embeddings (first run)...")

    embeddings = []

    for question in faq_questions:

        vector = get_embedding(question)
        embeddings.append(vector)

        time.sleep(0.7)  # avoid API rate limit

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    return np.array(embeddings)


faq_embeddings = load_or_create_embeddings()


# ====================================
# CHAT MEMORY
# ====================================

user_sessions = {}

# ====================================
# RATE LIMITER
# ====================================

last_ai_call = 0
AI_COOLDOWN = 2   # detik antar request AI

# anti spam user
user_last_message = {}
USER_COOLDOWN = 3


# ====================================
# RETRIEVE FAQ (EMBEDDING SEARCH)
# ====================================

def retrieve_faq(query, top_k=2):

    query_vector = get_embedding(query)

    similarities = []

    for vector in faq_embeddings:

        similarity = np.dot(query_vector, vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(vector)
        )

        similarities.append(similarity)

    similarities = np.array(similarities)

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    max_score = float(similarities[top_indices[0]])

    for idx in top_indices:

        results.append({
            "question": faq_data[idx]["question"],
            "answer": faq_data[idx]["answer"],
            "score": float(similarities[idx])
        })

    return results, max_score


# ====================================
# BUILD CONTEXT
# ====================================

def build_context(faqs):

    context = "Informasi resmi BKPSDM Kota Kendari:\n\n"

    for faq in faqs:

        context += f"Pertanyaan: {faq['question']}\n"
        context += f"Jawaban: {faq['answer']}\n\n"

    return context


# ====================================
# SYSTEM PROMPT
# ====================================

SYSTEM_PROMPT = """
Anda adalah chatbot layanan informasi BKPSDM Kota Kendari.

Tugas Anda membantu masyarakat mendapatkan informasi
terkait layanan kepegawaian seperti CPNS, PPPK,
mutasi, kenaikan pangkat, pensiun, dan administrasi ASN.

Gunakan bahasa Indonesia yang sopan dan jelas.

Utamakan informasi dari FAQ yang diberikan.
Jika informasi terbatas, gunakan pengetahuan umum
tentang administrasi ASN di Indonesia dan jelaskan
bahwa informasi dapat berubah sesuai kebijakan pemerintah.

Sarankan pengguna menghubungi BKPSDM Kota Kendari
untuk informasi resmi yang lebih lengkap.

Jika pertanyaan berada di luar layanan kepegawaian BKPSDM,
jawablah dengan sopan bahwa chatbot hanya melayani
informasi terkait layanan kepegawaian.
"""


# ====================================
# LOGGING PERTANYAAN
# ====================================

def save_log(user_id, question, answer):

    log_data = {
        "user": user_id,
        "question": question,
        "answer": answer
    }

    try:
        with open("logs.json", "r", encoding="utf-8") as f:
            logs = json.load(f)
    except:
        logs = []

    logs.append(log_data)

    with open("logs.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


# ====================================
# AUTO LEARNING FAQ
# ====================================

def analyze_logs():

    try:
        with open("logs.json", "r", encoding="utf-8") as f:
            logs = json.load(f)
    except:
        return []

    questions = [log["question"].lower() for log in logs]

    counter = Counter(questions)

    recommendations = []

    for question, count in counter.items():

        if count >= 3:

            recommendations.append({
                "question": question,
                "count": count
            })

    return recommendations

# ====================================
# Detect greeting
# ====================================

def detect_greeting(message):

    greetings = [
        "halo",
        "hai",
        "hi",
        "pagi",
        "siang",
        "sore",
        "malam",
        "assalamualaikum",
        "permisi",
        "halo min",
        "halo admin"
    ]

    msg = message.lower()

    for g in greetings:
        if g in msg:
            return True

    return False

# ====================================
# DETECT LINK / SPAM
# ====================================

def contains_link(text):

    text = text.lower()

    indicators = [
        "http://",
        "https://",
        "www.",
        ".com",
        ".id",
        ".net"
    ]

    for i in indicators:
        if i in text:
            return True

    return False

# ====================================
# NORMALIZE USER QUESTION
# ====================================

def normalize_question(question):

    prompt = f"""
    Ubah pertanyaan berikut menjadi pertanyaan yang lebih jelas,
    singkat, dan formal dalam Bahasa Indonesia tanpa
    mengubah makna.

    Pertanyaan:
    {question}

    Tulis hanya satu kalimat pertanyaan.
    """

    try:

        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:

        print("Normalize error:", e)

        return question

# ====================================
# SPLIT MULTI QUESTION
# ====================================

def split_questions(message):

    separators = [" dan ", " & ", "?", ";"]

    questions = [message]

    for sep in separators:

        temp = []

        for q in questions:

            if sep in q:
                parts = q.split(sep)

                for p in parts:
                    p = p.strip()

                    if p:
                        temp.append(p)

            else:
                temp.append(q)

        questions = temp

    # batasi maksimal 3 pertanyaan
    return questions[:3]

# ====================================
# AI RATE LIMITER
# ====================================

def wait_for_ai_slot():

    global last_ai_call

    now = time.time()

    diff = now - last_ai_call

    if diff < AI_COOLDOWN:
        time.sleep(AI_COOLDOWN - diff)

    last_ai_call = time.time()

# ====================================
# USER SPAM PROTECTION
# ====================================

def user_spam_protection(user_id):

    now = time.time()

    if user_id in user_last_message:

        diff = now - user_last_message[user_id]

        if diff < USER_COOLDOWN:
            return True

    user_last_message[user_id] = now

    return False

# ====================================
# CHAT ENDPOINT
# ====================================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.json

    user_id = data.get("user_id", "default_user")
    message = data.get("message")
    questions = split_questions(message)

    # anti spam protection
    if user_spam_protection(user_id):

        return jsonify({
            "reply": "Mohon tunggu beberapa detik sebelum mengirim pertanyaan berikutnya.",
            "similarity_score": 0,
            "retrieved_faq": []
        })
    
    if len(message) > 500:

        return jsonify({
            "reply": "Mohon kirim pertanyaan yang lebih singkat agar dapat diproses oleh sistem.",
            "similarity_score": 0,
            "retrieved_faq": []
        })


    # ====================================
    # MEMORY SESSION
    # ====================================

    if user_id not in user_sessions:
        user_sessions[user_id] = []

    history = user_sessions[user_id]

    # ====================================
    # GREETING DETECTION
    # ====================================

    if detect_greeting(message):

        reply = """
        Halo, selamat datang di layanan informasi BKPSDM Kota Kendari.

        Saya adalah chatbot yang dapat membantu memberikan informasi terkait layanan kepegawaian seperti:

        • CPNS
        • PPPK
        • Kenaikan Pangkat
        • Mutasi
        • Pensiun
        • Administrasi ASN

        Silakan ajukan pertanyaan Anda terkait layanan tersebut.
        """

        return jsonify({
            "reply": reply,
            "similarity_score": 0,
            "retrieved_faq": []
        })

    # ====================================
    # RETRIEVE FAQ (FIRST PASS)
    # ====================================

    answers = []
    highest_similarity = 0
    all_retrieved_faq = []

    for q in questions:

        relevant_faqs, similarity_score = retrieve_faq(q)

        if similarity_score > highest_similarity:
            highest_similarity = similarity_score

        all_retrieved_faq.extend(relevant_faqs)

        # jika sangat mirip FAQ
        if similarity_score > 0.85:

            answers.append(relevant_faqs[0]["answer"])
            continue

        # jika terlalu jauh
        if similarity_score < 0.35:

            answers.append(
                "Mohon maaf, kami belum menemukan informasi yang sesuai untuk pertanyaan tersebut."
            )
            continue

        # jika perlu AI
        context = build_context(relevant_faqs)

        prompt = f"""
    {SYSTEM_PROMPT}

    Informasi FAQ BKPSDM Kota Kendari:

    {context}

    Jawablah pertanyaan berikut dengan bahasa yang jelas dan sopan.

    Pertanyaan:
    {q}
    """

        wait_for_ai_slot()

        try:

            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt
            )

            answers.append(response.text)

        except:

            answers.append("Mohon maaf, sistem sedang sibuk. Silakan coba lagi.")

    reply = "\n\n".join(answers)

    # ====================================
    # NORMALIZE QUESTION IF NEEDED
    # ====================================

    if 0.35 < highest_similarity  < 0.5:

        normalized_question = normalize_question(message)

        relevant_faqs, similarity_score = retrieve_faq(normalized_question)

    # jika pertanyaan sangat mirip dengan FAQ
    if similarity_score > 0.85:

        reply = relevant_faqs[0]["answer"]

        save_log(user_id, message, reply)

        return jsonify({
            "reply": reply,
            "similarity_score": similarity_score,
            "retrieved_faq": relevant_faqs
        })

    # jika pertanyaan terlalu jauh dari FAQ
    if similarity_score < 0.35:

        reply = """
        Terima kasih telah menghubungi layanan informasi BKPSDM Kota Kendari.

        Saat ini kami belum menemukan informasi yang sesuai dengan pertanyaan Anda.

        Silakan ajukan pertanyaan terkait layanan kepegawaian seperti:
        • CPNS
        • PPPK
        • Kenaikan Pangkat
        • Mutasi
        • Pensiun
        • Administrasi ASN

        Untuk informasi lebih lanjut Anda juga dapat menghubungi BKPSDM Kota Kendari melalui kanal resmi.
        """

        return jsonify({
            "reply": reply,
            "similarity_score": similarity_score,
            "retrieved_faq": []
        })

    # ====================================
    # BUILD CONTEXT
    # ====================================

    context = build_context(relevant_faqs)


    # ====================================
    # PROMPT
    # ====================================

    prompt = f"""
    {SYSTEM_PROMPT}

    Informasi FAQ BKPSDM Kota Kendari:

    {context}

    Jawablah pertanyaan dengan bahasa yang jelas dan sopan.
    Gunakan maksimal 5–7 kalimat agar mudah dibaca pada WhatsApp.
    Jika informasi tidak sepenuhnya tersedia pada FAQ,
    gunakan pengetahuan umum tentang administrasi ASN
    di Indonesia dan sarankan pengguna menghubungi
    BKPSDM Kota Kendari untuk informasi resmi terbaru.

    Riwayat percakapan:
    {history}

    Pertanyaan pengguna:
    {message}

    
    """


    # ====================================
    # GEMINI RESPONSE (RETRY SYSTEM)
    # ====================================

    wait_for_ai_slot()

    reply = ""

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt
            )

            reply = response.text
            break

        except Exception as e:

            print("Gemini error:", e)
            time.sleep(2)

    if reply == "":
        reply = """
        Mohon maaf, layanan chatbot sedang mengalami kepadatan sistem.

        Silakan coba kembali beberapa saat lagi.

        Untuk informasi resmi Anda juga dapat menghubungi
        BKPSDM Kota Kendari melalui kanal layanan resmi.
        """

    # ====================================
    # SAVE MEMORY
    # ====================================

    history.append({
        "user": message,
        "bot": reply
    })

    if len(history) > 3:
        history.pop(0)


    # ====================================
    # SAVE LOG
    # ====================================

    save_log(user_id, message, reply)


    return jsonify({
        "reply": reply,
        "similarity_score": similarity_score,
        "retrieved_faq": relevant_faqs
    })

# ====================================
# SEND WHATSAPP
# ====================================

def send_whatsapp(target, message):
    token = os.getenv("FONNTE_TOKEN")

    url = "https://api.fonnte.com/send"

    payload = {
        "target": target,
        "message": message
    }

    headers = {
        "Authorization": token
    }

    # simulasi bot sedang mengetik
    time.sleep(1.5)

    requests.post(url, data=payload, headers=headers)


# ====================================
# AUTO LEARNING ENDPOINT
# ====================================

@app.route("/faq-suggestions", methods=["GET"])
def faq_suggestions():

    suggestions = analyze_logs()

    return jsonify({
        "recommended_faq": suggestions
    })

# ====================================
# WHATSAPP WEBHOOK
# ====================================

@app.route("/whatsapp", methods=["POST"])
def whatsapp():

    sender = request.form.get("sender")
    message = request.form.get("message")
    is_group = request.form.get("isGroup")
    message_type = request.form.get("type")

    # cegah bot membalas media
    if message_type and message_type != "text":
        return "OK"

    # cegah bot membalas pesan yang berisi link
    if message and contains_link(message):
        return "OK"

    # cegah bot membalas pesan sendiri (anti loop)
    if sender == "bot":
        return "OK"

    # cegah bot membalas pesan dari grup
    if is_group == "true":
        return "OK"

    # cegah pesan kosong
    if not message:
        return "OK"

    # kirim pesan ke chatbot AI
    response = app.test_client().post("/chat", json={
        "user_id": sender,
        "message": message
    })

    bot_reply = response.get_json()["reply"]

    send_whatsapp(sender, bot_reply)

    return "OK"

# ====================================
# RUN SERVER
# ====================================

if __name__ == "__main__":
    app.run(port=5000, debug=True)