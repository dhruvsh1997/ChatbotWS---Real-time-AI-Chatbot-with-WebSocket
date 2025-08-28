import pytest
from unittest.mock import patch, MagicMock
from app import app, socketio


class TestIntegration:
    """Integration tests for the complete chatbot workflow."""
    
    @patch('app.client_groq')
    def test_complete_chat_workflow(self, mock_groq):
        """Test complete chat workflow from connection to response."""
        # Mock Groq API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I'm doing great! How can I help you today?"
        mock_groq.chat.completions.create.return_value = mock_response
        
        # Create socket client
        client = socketio.test_client(app)
        
        # Step 1: Connect and get session
        received = client.get_received()
        
        # Find session_id event
        session_event = next((event for event in received if event['name'] == 'session_id'), None)
        assert session_event is not None
        session_id = session_event['args'][0]['session_id']
        
        # Find welcome message
        welcome_event = next((event for event in received if event['name'] == 'bot_message'), None)
        assert welcome_event is not None
        assert 'Hello there!' in welcome_event['args'][0]['message']
        
        # Step 2: Send user message
        client.emit('user_message', {
            'session_id': session_id,
            'message': 'How are you doing?'
        })
        
        # Step 3: Verify AI response
        received = client.get_received()
        bot_response = next((event for event in received if event['name'] == 'bot_message'), None)
        assert bot_response is not None
        assert "I'm doing great!" in bot_response['args'][0]['message']
        
        # Step 4: Verify Groq API was called correctly
        mock_groq.chat.completions.create.assert_called_once()
        call_args = mock_groq.chat.completions.create.call_args
        
        # Check model
        assert call_args.kwargs['model'] == 'llama-3.3-70b-versatile'
        
        # Check messages structure
        messages = call_args.kwargs['messages']
        assert len(messages) == 2  # system prompt + user message
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == 'How are you doing?'
        
        # Step 5: Test conversation continuity
        client.emit('user_message', {
            'session_id': session_id,
            'message': 'What is the weather like?'
        })
        
        # Should be called twice now
        assert mock_groq.chat.completions.create.call_count == 2
        
        # Check that conversation history is maintained
        second_call_args = mock_groq.chat.completions.create.call_args
        messages = second_call_args.kwargs['messages']
        assert len(messages) == 4  # system + user1 + assistant1 + user2
        
        client.disconnect()
    
    def test_multiple_concurrent_sessions(self):
        """Test handling multiple concurrent user sessions."""
        client1 = socketio.test_client(app)
        client2 = socketio.test_client(app)
        
        # Get session IDs
        received1 = client1.get_received()
        received2 = client2.get_received()
        
        session1_event = next((event for event in received1 if event['name'] == 'session_id'), None)
        session2_event = next((event for event in received2 if event['name'] == 'session_id'), None)
        
        session1_id = session1_event['args'][0]['session_id']
        session2_id = session2_event['args'][0]['session_id']
        
        # Sessions should be different
        assert session1_id != session2_id
        
        # Both should have conversation histories
        from app import conversation_histories
        assert session1_id in conversation_histories
        assert session2_id in conversation_histories
        
        client1.disconnect()
        client2.disconnect()
    
    def test_error_handling_workflow(self):
        """Test error handling in the complete workflow."""
        client = socketio.test_client(app)
        
        # Test with invalid session ID
        client.emit('user_message', {
            'session_id': 'invalid-session-id',
            'message': 'Hello'
        })
        
        received = client.get_received()
        error_event = next((event for event in received if event['name'] == 'error'), None)
        assert error_event is not None
        assert 'Session not found' in error_event['args'][0]['message']
        
        client.disconnect()