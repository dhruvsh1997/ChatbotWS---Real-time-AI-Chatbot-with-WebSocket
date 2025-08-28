import pytest
import os
from unittest.mock import patch, MagicMock
from app import app, socketio, get_timestamp, conversation_histories


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-secret-key'
    with app.test_client() as client:
        yield client


@pytest.fixture
def socket_client():
    """Create a test client for Socket.IO."""
    return socketio.test_client(app)


class TestFlaskRoutes:
    """Test Flask HTTP routes."""
    
    def test_index_route(self, client):
        """Test the main index route returns chat.html."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Playschool Chatbot' in response.data
        assert b'socket.io' in response.data


class TestSocketEvents:
    """Test Socket.IO events."""
    
    def test_connect_event(self, socket_client):
        """Test WebSocket connection establishes properly."""
        received = socket_client.get_received()
        
        # Should receive session_id and welcome message
        assert len(received) >= 2
        
        # Check for session_id event
        session_event = next((event for event in received if event['name'] == 'session_id'), None)
        assert session_event is not None
        assert 'session_id' in session_event['args'][0]
        
        # Check for welcome message
        welcome_event = next((event for event in received if event['name'] == 'bot_message'), None)
        assert welcome_event is not None
        assert 'Hello there!' in welcome_event['args'][0]['message']
    
    @patch('app.client_groq')
    def test_user_message_success(self, mock_groq, socket_client):
        """Test successful user message processing."""
        # Mock Groq API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        mock_groq.chat.completions.create.return_value = mock_response
        
        # Get session ID from connection
        received = socket_client.get_received()
        session_event = next((event for event in received if event['name'] == 'session_id'), None)
        session_id = session_event['args'][0]['session_id']
        
        # Send user message
        socket_client.emit('user_message', {
            'session_id': session_id,
            'message': 'Hello chatbot!'
        })
        
        # Check response
        received = socket_client.get_received()
        bot_response = next((event for event in received if event['name'] == 'bot_message'), None)
        assert bot_response is not None
        assert 'Hello! How can I help you?' in bot_response['args'][0]['message']
    
    def test_user_message_invalid_session(self, socket_client):
        """Test user message with invalid session ID."""
        socket_client.emit('user_message', {
            'session_id': 'invalid-session',
            'message': 'Hello chatbot!'
        })
        
        # Should receive error
        received = socket_client.get_received()
        error_event = next((event for event in received if event['name'] == 'error'), None)
        assert error_event is not None
        assert 'Session not found' in error_event['args'][0]['message']
    
    @patch('app.client_groq')
    def test_user_message_api_error(self, mock_groq, socket_client):
        """Test user message when Groq API fails."""
        # Mock API error
        mock_groq.chat.completions.create.side_effect = Exception("API Error")
        
        # Get session ID from connection
        received = socket_client.get_received()
        session_event = next((event for event in received if event['name'] == 'session_id'), None)
        session_id = session_event['args'][0]['session_id']
        
        # Send user message
        socket_client.emit('user_message', {
            'session_id': session_id,
            'message': 'Hello chatbot!'
        })
        
        # Check error response
        received = socket_client.get_received()
        bot_response = next((event for event in received if event['name'] == 'bot_message'), None)
        assert bot_response is not None
        assert 'Sorry, I encountered an error' in bot_response['args'][0]['message']


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_timestamp_format(self):
        """Test timestamp format is HH:MM."""
        timestamp = get_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) == 5  # HH:MM format
        assert ':' in timestamp
        
        # Should be valid time format
        hours, minutes = timestamp.split(':')
        assert 0 <= int(hours) <= 23
        assert 0 <= int(minutes) <= 59


class TestConfiguration:
    """Test application configuration."""
    
    def test_environment_variables(self):
        """Test that required environment variables are handled."""
        # This test assumes GROQ_API_KEY is set in the environment
        # In actual deployment, this would be required
        groq_key = os.environ.get('GROQ_API_KEY')
        # In testing environment, this might not be set, so we just check the handling
        assert groq_key is not None or os.environ.get('TESTING') == 'True'
    
    def test_app_configuration(self):
        """Test Flask app configuration."""
        assert app.config.get('TESTING') is not None
        assert app.config.get('SECRET_KEY') is not None


class TestSessionManagement:
    """Test conversation session management."""
    
    def test_conversation_history_storage(self, socket_client):
        """Test that conversation histories are stored properly."""
        initial_count = len(conversation_histories)
        
        # Connect to create new session
        received = socket_client.get_received()
        session_event = next((event for event in received if event['name'] == 'session_id'), None)
        session_id = session_event['args'][0]['session_id']
        
        # Should have one more session
        assert len(conversation_histories) == initial_count + 1
        assert session_id in conversation_histories
        
        # Should have system prompt in history
        history = conversation_histories[session_id]
        assert len(history) == 1
        assert history[0]['role'] == 'system'
        assert 'helpful AI assistant' in history[0]['content']