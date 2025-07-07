# AI Voice Receptionist: A Conversational Voice AI with LangChain & Twilio

## 🚀 Demo Video

[![AI Voice Assistant Portfolio Demo](images/AI_receptionist.gif)](https://vimeo.com/1099317316)

**[Watch the full demo on Vimeo](https://vimeo.com/1099317316)**

*(Caption: This video demonstrates the AI's core capabilities: answering knowledge-based questions, scheduling an appointment, proactively offering an SMS confirmation, seamlessly switching to DTMF keypad entry for phone number capture, and successfully sending the confirmation text.)*

---
## ✨ Features

* **Real-time Telephony Interface:**
    * Handles incoming phone calls via a **Twilio** phone number.
    * Uses **TwiML** (Twilio Markup Language) to dynamically control the call flow.
    * Generates responses using high-quality, human-like text-to-speech (**OpenAI TTS**).

* **Intelligent Task Execution (LangChain Agent):**
    * A central **AgentExecutor** orchestrates the entire conversation.
    * The Agent intelligently chooses from a suite of tools to fulfill user requests:
        * `schedule_appointment`: Schedules appointments.
        * `send_sms`: Sends SMS messages using the Twilio REST API.
        * `get_current_weather`: Fetches real-time weather data.
        * `convert_currency`: Performs currency conversions.
        * `search_knowledge_base`: A RAG tool to answer questions about the business.

* **Knowledge Base Q&A (RAG):**
    * Answers questions based on a pre-loaded knowledge document (`knowledge.txt`).
    * Uses **ChromaDB** for efficient, local vector storage and retrieval.

* **Advanced Conversational Design:**
    * **Proactive Behavior:** After scheduling an appointment, the agent doesn't just stop—it actively asks the user if they'd like a confirmation, demonstrating next-level intelligence.
    * **Hybrid Input (Voice & DTMF):** To solve the problem of inaccurate speech recognition for numbers, the system automatically redirects the user to a secure **DTMF** keypad entry mode, ensuring 100% accuracy for critical data.
    * **Asynchronous Backend:** Prevents call drops during long AI processing times by immediately acknowledging the user's query, processing the request in a **background thread**, and then redirecting the live call to deliver the final answer.

---
## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Telephony & SMS:** Twilio (Voice API, TwiML, REST API)
* **AI / LLM Framework:** LangChain (Agents, Tools, Chains)
* **LLM & TTS:** OpenAI (`gpt-4o`, `tts-1-hd`)
* **Vector Store (for RAG):** ChromaDB
* **Key Libraries:** `twilio`, `langchain`, `openai`, `chromadb`, `phonenumbers`, `python-dotenv`
* **Development:** Git, GitHub, Ngrok, Virtual Environment (`.venv`)

---
## 🔑 Key Architectural Decisions & Learnings

* **Building a Non-Blocking Voice Experience:** The biggest challenge in voice AI is latency. I implemented a multi-threaded architecture where user queries are handled in a background thread. This allows the main application to immediately return a "holding message" to Twilio, keeping the call alive while the Agent thinks. Once a response is ready, the live call is updated using the Twilio REST API, creating a seamless user experience.

* **Designing a Proactive Agent:** Through careful **prompt engineering**, I instructed the LangChain agent to do more than just react. It now anticipates user needs, such as offering an SMS confirmation after an appointment is booked, making the interaction feel more like talking to a human assistant.

* **Solving Data-Entry Accuracy with DTMF:** Early tests showed that relying on speech-to-text for phone numbers was unreliable. I engineered a solution where the agent, upon needing a number, redirects the call to a dedicated DTMF-only input flow. This hybrid approach uses the best input method for each task, dramatically increasing the system's reliability.

---
## ⚙️ Setup and Usage (Local)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yskmtb0714/ai-voice-receptionist.git](https://github.com/yskmtb0714/ai-voice-receptionist.git)
    cd ai-voice-receptionist
    ```
2.  **Backend Setup:**
    * Create and activate a Python virtual environment.
    * Install dependencies: `pip install -r requirements.txt`
    * Create a `.env` file and add your API keys for `OPENAI`, `TWILIO`, etc.

3.  **Run the Application:**
    * **Terminal 1 (Flask App):** `python app.py`
    * **Terminal 2 (Ngrok):** `ngrok http 5000` (to expose your local server)

4.  **Configure Twilio:**
    * Copy the `https` URL from your Ngrok terminal.
    * In the Twilio Console, go to your phone number's settings.
    * Under "A CALL COMES IN", paste the Ngrok URL followed by `/voice`.

5.  **Test:**
    * Call your Twilio phone number and start talking to your AI assistant!

---
## 🔮 Future Work

* Integrate with a real calendar API (Google Calendar) to make `schedule_appointment` fully functional.
* Implement persistent, user-specific chat history using a database like SQLite or Redis.
* Explore real-time audio streaming (using Twilio Media Streams) to further reduce latency.