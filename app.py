# flask>=0.68.0
# flask-socketio>=5.0.0
# groq>=0.3.0
# python-dotenv>=0.19.0

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os
from groq import Groq
import uuid
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('GROQ_API_KEY')
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize Groq client
# Make sure to set your GROQ_API_KEY environment variable
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

client_groq = Groq(api_key=GROQ_API_KEY)

# Store conversation histories by session ID
conversation_histories = {}

# System prompt for the chatbot
SYSTEM_PROMPT = "You are a helpful AI assistant with a friendly and playful personality. Answer questions concisely and accurately."

@app.route('/')
def index():
    """Render the chat page"""
    return render_template('chat.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    session_id = str(uuid.uuid4())
    conversation_histories[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Send session ID to client
    emit('session_id', {'session_id': session_id})
    
    # Send welcome message
    emit('bot_message', {
        'message': "Hello there! I'm your friendly chatbot. How can I help you today? 😊",
        'timestamp': get_timestamp()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('user_message')
def handle_user_message(data):
    """Handle user message and generate response"""
    session_id = data.get('session_id')
    message = data.get('message')
    
    if not session_id or session_id not in conversation_histories:
        emit('error', {'message': 'Session not found'})
        return
    
    # Add user message to conversation history
    conversation_histories[session_id].append({"role": "user", "content": message})
    
    try:
        # Generate response using Groq API
        response = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=conversation_histories[session_id]
        )
        
        # Extract response
        bot_response = response.choices[0].message.content
        
        # Add bot response to conversation history
        conversation_histories[session_id].append({"role": "assistant", "content": bot_response})
        
        # Send response to client
        emit('bot_message', {
            'message': bot_response,
            'timestamp': get_timestamp()
        })
        
    except Exception as e:
        emit('bot_message', {
            'message': f"Sorry, I encountered an error: {str(e)}",
            'timestamp': get_timestamp()
        })

def get_timestamp():
    """Get current timestamp in HH:MM format"""
    from datetime import datetime
    return datetime.now().strftime('%H:%M')

if __name__ == '__main__':
    socketio.run(app, debug=True)