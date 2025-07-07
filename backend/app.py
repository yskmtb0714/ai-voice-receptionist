import os
import threading
import traceback
from typing import Optional
from datetime import datetime

import requests
import phonenumbers
from twilio.base.exceptions import TwilioRestException
from dotenv import load_dotenv
from flask import Flask, request, url_for
from openai import OpenAI
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather

# LangChainのコンポーネントをインポート
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import AIMessage, HumanMessage

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAIクライアントを初期化
try:
    openai_client = OpenAI()
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    openai_client = None

# Flaskアプリケーションを初期化
app = Flask(__name__)

# --- Agentが使用するツールの定義 ---

@tool
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Gets the current real-time weather for a specified location."""
    print(f"--- Tool: get_current_weather(location='{location}', unit='{unit}') called ---")
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key: return "Error: Weather API key is not configured."
    units_for_api = "imperial" if unit and unit.lower() == "fahrenheit" else "metric"
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": api_key, "units": units_for_api, "lang": "en"}
    try:
        api_response = requests.get(base_url, params=params, timeout=10)
        api_response.raise_for_status()
        data = api_response.json()
        if data.get("cod") != 200: return f"Error from weather API: {data.get('message', 'Unknown error')}"
        main_data, weather_data, city_name = data.get("main"), data.get("weather")[0] if data.get("weather") else {}, data.get("name")
        if not (main_data and weather_data and city_name): return f"Error: Could not parse weather data for {location}."
        temp, description, temp_unit_str = main_data.get("temp"), weather_data.get("description"), "degrees Celsius" if units_for_api == "metric" else "degrees Fahrenheit"
        return f"The current weather in {city_name} is {description}, with a temperature of {temp} {temp_unit_str}."
    except Exception as e:
        print(f"!!! Error in get_current_weather tool: {e}")
        return f"Sorry, I couldn't retrieve the weather for {location} right now."

@tool
def schedule_appointment(date: str, time: str, name: str, topic: str) -> str:
    """Schedules an appointment. After this tool is used, the system will ask about sending an SMS confirmation."""
    print(f"--- Tool: schedule_appointment called with date='{date}', time='{time}', name='{name}', topic='{topic}' ---")
    try:
        confirmation_message = f"Okay, I have successfully scheduled an appointment for {name} on {date} at {time} to discuss '{topic}'."
        print(f"Appointment confirmation generated: {confirmation_message}")
        # 予約確認メッセージ（SMSの本文になる）のみを返す
        return confirmation_message
    except Exception as e:
        print(f"!!! Error in schedule_appointment tool: {e}")
        return "Sorry, I encountered an issue while trying to schedule the appointment."

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts a specified amount of money from one currency to another using real-time exchange rates."""
    print(f"--- Tool: convert_currency(amount={amount}, from={from_currency}, to={to_currency}) called ---")
    api_key = os.getenv("EXCHANGERATE_API_KEY")
    if not api_key: return "Error: Currency conversion API key is not configured."
    try:
        amount_float, from_curr, to_curr = float(amount), str(from_currency).upper(), str(to_currency).upper()
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_curr}/{to_curr}/{amount_float}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("result") == "success":
            return f"{amount_float:.2f} {from_curr} is approximately {data['conversion_result']:.2f} {to_curr}."
        else:
            return "Sorry, I could not perform the currency conversion due to an API issue."
    except Exception as e:
        print(f"!!! Error in convert_currency tool: {e}")
        return f"Sorry, an error occurred during currency conversion."

@tool
def send_sms(to: str, body: str) -> str:
    """Sends an SMS message to a specified phone number. The 'to' parameter must be a valid E.164 formatted phone number (e.g., +1234567890). This tool performs validation internally."""
    print(f"--- Tool: send_sms(to='{to}', body='{body}') called ---")

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_phone_number = os.getenv("TWILIO_PHONE_NUMBER")

    if not all([account_sid, auth_token, from_phone_number]):
        return "Error: Twilio service is not configured on the server."

    try:
        # 1. 電話番号のバリデーション
        parsed_number = phonenumbers.parse(to, None)
        if not phonenumbers.is_valid_number(parsed_number):
            return f"Error: The provided phone number '{to}' is not in a valid format."

        e164_number = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
        print(f"Validated E.164 number: {e164_number}")

        # 2. Twilio APIを呼び出してSMSを送信
        client = Client(account_sid, auth_token)
        message = client.messages.create(body=body, from_=from_phone_number, to=e164_number)
        print(f"SMS sent successfully. SID: {message.sid}")
        return f"An SMS message has been successfully sent to {e164_number}."
    except phonenumbers.phonenumberutil.NumberParseException:
        return f"Error: The phone number '{to}' could not be parsed. Please ensure it is a valid number with a country code."
    except TwilioRestException as e:
        print(f"!!! Twilio API Error in send_sms tool: {e}")
        if e.code == 21211:  # Invalid 'To' Phone Number
            return f"Error: The phone number '{to}' is invalid according to Twilio."
        elif e.code == 20003:  # Authentication Error
            return "Error: Twilio authentication failed. Please check server configuration."
        return f"Error: A Twilio API error occurred: {e.msg}"
    except Exception as e:
        print(f"!!! An unexpected error occurred in send_sms tool: {e}")
        return "Error: An unexpected server error occurred while trying to send the SMS."

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Sends an email to a specified address."""
    print(f"--- Tool: send_email(to='{to}', subject='{subject}', body='{body}') called ---")
    try:
        confirmation_message = f"An email with the subject '{subject}' has been successfully sent to {to}."
        print(f"Simulating email sent: {confirmation_message}")
        return confirmation_message
    except Exception as e:
        print(f"!!! Error in send_email tool: {e}")
        return "Sorry, I encountered an issue while trying to send the email."

def setup_rag_retriever():
    """RAG: Loads and prepares the knowledge base for searching."""
    print("--- Setting up RAG retriever ---")
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        print("Loading existing vector store from disk.")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    else:
        print("Creating new vector store and persisting to disk.")
        loader = TextLoader("./knowledge.txt", encoding="utf-8")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=persist_directory)
        print("Vector store created and persisted successfully.")
    retriever = vectorstore.as_retriever()
    print("Retriever setup complete.")
    return retriever

# --- Agentのセットアップ ---
agent_executor = None
call_sessions = {}  # 通話ごとのセッション情報を管理

def initialize_agent():
    """Initializes the main Agent Executor with tools and custom prompt."""
    global agent_executor
    print("--- Initializing Agent Executor with custom prompt for DTMF redirect ---")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    retriever = setup_rag_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "search_ai_cafe_knowledge",
        "Searches and returns information about the AI Assistant Cafe...",
    )
    tools = [
        get_current_weather,
        schedule_appointment,
        convert_currency,
        send_sms,
        send_email,
        retriever_tool,
    ]
    
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # ★★★ 現在の日付をプロンプトに追加 ★★★
    current_date_str = datetime.now().strftime("%A, %B %d, %Y")
    date_instruction = (
        f"\n\n--- CONTEXTUAL INFORMATION ---\n"
        f"The current date is {current_date_str}. You MUST use this for any date-related queries like 'today' or 'tomorrow'."
    )

    # ★★★ Agentの行動ルールをDTMFリダイレクト用に更新 ★★★
    new_instruction = (
        "\n\n--- IMPORTANT RULES ---\n"
        "1. After the 'schedule_appointment' tool returns a confirmation message (e.g., 'Okay, I have successfully scheduled...'), "
        "your response to the user MUST be constructed in two parts: first, repeat the exact confirmation message from the tool. "
        "Second, append the question 'Would you like me to send a confirmation via SMS?'"
        "\n\n"
        "2. If the user agrees to receive an SMS (e.g., says 'yes', 'okay', 'sure'), your ONLY job is to respond with the exact, single keyword `REDIRECT_TO_NUMBER_ENTRY`. "
        "Do not say anything else, just return that keyword. The system will handle the phone number entry process."
    )
    try:
        prompt.messages[0].prompt.template += date_instruction + new_instruction
        print("Successfully appended new instruction to the agent prompt.")
    except Exception as e:
        print(f"Could not modify prompt, using original. Error: {e}")

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=7,
    )
    print(f"Agent Executor initialized with {len(tools)} tools.")

# --- Webhookルート定義 ---

CONTEXTUAL_HINTS = "AI Assistant Cafe, Yirgacheffe, Algorithm Cheesecake, Brisbane, appointment, schedule, weather, currency"
NUMERIC_HINTS = "+1,+61,+81,0,1,2,3,4,5,6,7,8,9"
COMBINED_HINTS = f"{CONTEXTUAL_HINTS}, {NUMERIC_HINTS}"

# /voice ルートを修正

@app.route("/voice", methods=['GET', 'POST'])
def voice():
    response = VoiceResponse()
    gather = Gather(
        input='speech',
        action='/handle-gather',
        method='POST',
        speech_timeout='6',  # ★★★ 最初の待ち時間を6秒に設定 ★★★
        language='en-US',
        speech_model='phone_call',
        hints=COMBINED_HINTS
    )
    gather.say("Hello, this is your AI Voice Assistant. How may I help you today?", voice='Polly.Joanna')
    response.append(gather)
    response.redirect('/voice', method='POST')
    return str(response)

@app.route("/handle-gather", methods=['GET', 'POST'])
def handle_gather():
    response = VoiceResponse()

    if 'SpeechResult' in request.values:
        transcribed_text = request.values['SpeechResult'].lower()
        call_sid = request.values.get('CallSid') # CallSidを早めに取得

        # --- 高速パス（Fast Path）の定義 ---
        GREETING_KEYWORDS = ["thank you", "thanks", "thank u"]
        CLOSING_KEYWORDS = ["bye", "goodbye"]

        # 感謝の言葉が聞こえた場合の即時応答
        if any(keyword in transcribed_text for keyword in GREETING_KEYWORDS):
            print("Fast Path: Detected a greeting. Replying immediately.")
            gather = Gather(
                input='speech', action='/handle-gather', method='POST',
                speech_timeout='auto', language='en-US', speech_model='phone_call',
                hints=COMBINED_HINTS
            )
            gather.say("You're welcome! Is there anything else I can help with?", voice='Polly.Joanna')
            response.append(gather)
            response.redirect('/voice', method='POST')
            return str(response)

        # 終了の言葉が聞こえた場合の即時応答
        elif any(keyword in transcribed_text for keyword in CLOSING_KEYWORDS):
            print("Fast Path: Detected a closing phrase. Ending the call.")
            if call_sid and call_sid in call_sessions:
                del call_sessions[call_sid]
                print(f"Cleared session for CallSid: {call_sid}")
            response.say("Goodbye!", voice='Polly.Joanna')
            response.hangup()
            return str(response)
            
        # --- 低速パス（Slow Path） ---
        # 上記の単純なキーワード以外は、すべてAgentに渡す
        else:
            print("Slow Path: Passing complex query to the agent.")
            if call_sid not in call_sessions:
                call_sessions[call_sid] = {"chat_history": []}
                print(f"Created new session for CallSid: {call_sid}")
            
            host_url = request.host_url
            thread = threading.Thread(
                target=process_agent_and_update_call,
                args=(transcribed_text, call_sid, host_url)
            )
            thread.start()
            
            # SMS確認への応答を待っている状況かどうかをチェック
            session = call_sessions.get(call_sid, {})
            if session.get('expecting_sms_confirmation'):
                print("Context: Expecting SMS confirmation. Skipping the 'One moment' message and just pausing.")
                # フラグを消費したのでリセット
                session['expecting_sms_confirmation'] = False
                response.pause(length=30) # Agentがリダイレクト処理を行うのを待つ
            else:
                response.say("Got it. One moment while I look that up for you.", voice='Polly.Joanna')
                response.pause(length=30)

    else:
        # 音声入力がなかった場合の処理
        response.say("I'm sorry, I didn't catch that. Could you please say that again?", voice='Polly.Joanna')
        response.redirect('/voice', method='POST')

    return str(response)

def process_agent_and_update_call(transcribed_text, call_sid, host_url):
    """Handles the main agent logic in a background thread."""
    global agent_executor
    try:
        session_history = call_sessions.get(call_sid, {}).get("chat_history", [])
        agent_response = agent_executor.invoke({"input": transcribed_text, "chat_history": session_history})
        ai_response_text = agent_response.get("output")

        # ★★★ Agentがリダイレクトを要求したかチェック ★★★
        if "REDIRECT_TO_NUMBER_ENTRY" in ai_response_text:
            print("Agent requested redirect to number entry.")
            # デフォルトのSMS本文
            sms_body = "Your appointment is confirmed."
            if session_history and isinstance(session_history[-1], AIMessage):
                full_response = session_history[-1].content
                # SMSの質問部分を区切り文字として、それより前の部分を本文として抽出
                separator = "Would you like me to send a confirmation via SMS?"
                if separator in full_response:
                    sms_body = full_response.split(separator)[0].strip()
                else:
                    sms_body = full_response.strip() # フォールバック
            call_sessions[call_sid]['sms_body'] = sms_body
            print(f"Saved SMS body to session: '{sms_body}'")
            redirect_url = host_url + "gather-phone-number"
            client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
            client.calls(call_sid).update(url=redirect_url, method='POST')
            return # スレッドを終了

        # 通常の応答処理
        session_history.append(HumanMessage(content=transcribed_text))
        session_history.append(AIMessage(content=ai_response_text))
        call_sessions[call_sid]["chat_history"] = session_history

        speech_file_path = os.path.join(os.path.dirname(__file__), 'static', 'response.mp3')
        with openai_client.audio.speech.with_streaming_response.create(model="tts-1-hd", voice="nova", input=ai_response_text) as tts_streaming_response:
            tts_streaming_response.stream_to_file(speech_file_path)

        response_url = host_url + "play_final_response"
        client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        client.calls(call_sid).update(url=response_url, method='POST')
        print(f"Redirecting call {call_sid} to play the response.")
    except Exception as e:
        print(f"Error in background thread: {e}")
        traceback.print_exc()

@app.route("/play_final_response", methods=['GET', 'POST'])
def play_final_response():
    response = VoiceResponse()
    audio_file_url = request.host_url + url_for('static', filename='response.mp3')
    
    # ★★★ 現在の会話の文脈を取得 ★★★
    call_sid = request.values.get('CallSid')
    last_response_text = ""
    # セッション情報から、直前のAIの応答内容を取得
    if call_sid and call_sid in call_sessions and call_sessions[call_sid]['chat_history']:
        # chat_historyの最後の要素がAIMessageであることを確認
        if isinstance(call_sessions[call_sid]['chat_history'][-1], AIMessage):
            last_response_text = call_sessions[call_sid]['chat_history'][-1].content.lower()

    # ★★★ 状況に応じたヒントを動的に生成 ★★★
    current_hints = COMBINED_HINTS
    # セッションが存在する場合、次の入力がSMS確認への応答であるという期待状態をリセット
    if call_sid and call_sid in call_sessions:
        call_sessions[call_sid]['expecting_sms_confirmation'] = False

    # 直前のAIの応答がSMS確認の質問だった場合、肯定的なヒントを追加
    if "sms" in last_response_text and "confirmation" in last_response_text:
        print("Adding affirmative hints for SMS confirmation.")
        affirmative_hints = "yes, yes please, sure, okay, that's correct, right"
        current_hints = f"{affirmative_hints}, {COMBINED_HINTS}"
        # 次のユーザー入力がSMS確認への応答であると期待するフラグを立てる
        if call_sid and call_sid in call_sessions:
            call_sessions[call_sid]['expecting_sms_confirmation'] = True

    # ★★★ 動的に生成したヒントを使ってGatherを設定 ★★★
    gather = Gather(
        input='speech',
        action='/handle-gather',
        method='POST',
        speech_timeout='auto',
        language='en-US',
        speech_model='phone_call',
        hints=current_hints # 動的に生成したヒントを使用
    )
    gather.play(audio_file_url)
    response.append(gather)
    response.redirect('/voice', method='POST')
    return str(response)
# ★★★ DTMF入力用の新しいルート ★★★
@app.route("/gather-phone-number", methods=['GET', 'POST'])
def gather_phone_number():
    """Generates TwiML to gather phone number via DTMF."""
    response = VoiceResponse()
    gather = Gather(
        input='dtmf', action='/handle-dtmf-input', method='POST',
        finish_on_key='#', timeout=15
    )
    gather.say(
        "To receive an SMS confirmation, please use your phone's keypad to enter the phone number, "
        "starting with the country code. For example, for Australia, start with 6 1. "
        "Press the pound key when you are finished.",
        voice='Polly.Joanna'
    )
    response.append(gather)
    response.say("Sorry, I didn't receive any input. Goodbye.", voice='Polly.Joanna')
    response.hangup()
    return str(response)

# 修正後の /handle-dtmf-input ルート

@app.route("/handle-dtmf-input", methods=['GET', 'POST'])
def handle_dtmf_input():
    """入力されたDTMF番号を処理し、SMSを送信するルート"""
    response = VoiceResponse()
    call_sid = request.values['CallSid']
    digits = request.values.get('Digits')
    
    if not digits:
        response.say("No number was entered. Please try again later. Goodbye.", voice='Polly.Joanna')
        response.hangup()
        return str(response)

    e164_number = f"+{digits}"
    print(f"Received DTMF digits: {digits}, formatted as: {e164_number}")

    sms_body = call_sessions.get(call_sid, {}).get('sms_body', "This is your confirmation.")
    
    try:
        send_result = send_sms.invoke({
            "to": e164_number,
            "body": sms_body
        })
        
        if "Error:" in send_result:
            print(f"SMS sending failed with message: {send_result}")
            # エラーメッセージの内容に応じて応答を分岐
            if "authentication failed" in send_result:
                response.say("I'm sorry, there's a configuration issue on our end, and I couldn't send the SMS. Goodbye.", voice='Polly.Joanna')
            elif "invalid" in send_result.lower() or "not in a valid format" in send_result or "could not be parsed" in send_result:
                response.say("Sorry, there was an issue with the number you entered. It seems to be invalid. Please call back to try again. Goodbye.", voice='Polly.Joanna')
            else:
                response.say("I'm sorry, an unexpected error occurred, and I couldn't send the SMS. Goodbye.", voice='Polly.Joanna')
        else:
            # SMS送信に成功
            response.say(f"Thank you. A confirmation SMS has been sent to the number you provided. Goodbye.", voice='Polly.Joanna')
            
    except Exception as e:
        print(f"An unexpected error occurred while invoking send_sms tool: {e}")
        response.say("I'm sorry, an internal error occurred while trying to send the message. Goodbye.", voice='Polly.Joanna')


    response.hangup()
    
    if call_sid in call_sessions:
        del call_sessions[call_sid]
        print(f"Cleared session for CallSid: {call_sid}")

    return str(response)

if __name__ == "__main__":
    initialize_agent()
    print(">>> Starting Flask server for AI Voice Receptionist...")
    app.run(debug=True, port=5000, host='0.0.0.0')