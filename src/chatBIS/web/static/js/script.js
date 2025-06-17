document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');

    // Session management
    let sessionId = localStorage.getItem('chatSessionId') || null;

    // Function to add a message to the chat
    function addMessage(content, isUser = false) {
        console.log('addMessage called with:', { content: content.substring(0, 100) + '...', isUser });

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        if (isUser) {
            // For user messages, keep as plain text
            const messageParagraph = document.createElement('p');
            messageParagraph.textContent = content;
            messageContent.appendChild(messageParagraph);
        } else {
            // For assistant messages, create a div to hold formatted content
            const messageDiv = document.createElement('div');
            messageDiv.className = 'markdown-content';

            // Enhanced markdown-like formatting
            let formattedContent = content
                // Headers (must be processed before line breaks)
                .replace(/^### (.*$)/gm, '<h3>$1</h3>')           // H3 headers
                .replace(/^## (.*$)/gm, '<h2>$1</h2>')            // H2 headers
                .replace(/^# (.*$)/gm, '<h1>$1</h1>')             // H1 headers
                // Text formatting
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
                .replace(/\*(.*?)\*/g, '<em>$1</em>')             // Italic
                .replace(/`(.*?)`/g, '<code>$1</code>')           // Inline code
                // Line breaks (convert to proper paragraphs and lists)
                .replace(/\n\n/g, '</p><p>')                      // Double line breaks = new paragraph
                .replace(/\n/g, '<br>');                          // Single line breaks = <br>

            // Handle lists with simple regex (basic support)
            // Convert bullet points to proper list items
            formattedContent = formattedContent.replace(/((?:- .*<br>)+)/g, function(match) {
                const items = match.replace(/- (.*?)<br>/g, '<li>$1</li>');
                return '<ul>' + items + '</ul>';
            });

            // Convert numbered lists to proper list items
            formattedContent = formattedContent.replace(/((?:\d+\. .*<br>)+)/g, function(match) {
                const items = match.replace(/\d+\. (.*?)<br>/g, '<li>$1</li>');
                return '<ol>' + items + '</ol>';
            });

            // Wrap in paragraph tags if not already wrapped
            if (!formattedContent.includes('<h') && !formattedContent.includes('<ul') && !formattedContent.includes('<ol')) {
                formattedContent = '<p>' + formattedContent + '</p>';
            }

            messageDiv.innerHTML = formattedContent;
            messageContent.appendChild(messageDiv);

            console.log('Assistant message added with content:', formattedContent.substring(0, 100) + '...');
        }

        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);

        // Scroll to the bottom of the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to add a loading indicator
    function addLoadingIndicator() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant';
        loadingDiv.id = 'loading-message';

        const loadingContent = document.createElement('div');
        loadingContent.className = 'loading';

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            loadingContent.appendChild(dot);
        }

        loadingDiv.appendChild(loadingContent);
        chatMessages.appendChild(loadingDiv);

        // Scroll to the bottom of the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to remove the loading indicator
    function removeLoadingIndicator() {
        const loadingMessage = document.getElementById('loading-message');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }

    // Function to send a message to the server
    async function sendMessage(message) {
        try {
            // Add loading indicator
            addLoadingIndicator();

            // Prepare request body with session management
            const requestBody = {
                query: message,
                session_id: sessionId
            };

            // Send the message to the server
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            // Remove loading indicator
            removeLoadingIndicator();

            // Parse the response
            const data = await response.json();

            if (data.success) {
                // Update session ID if provided
                if (data.session_id) {
                    sessionId = data.session_id;
                    localStorage.setItem('chatSessionId', sessionId);
                }

                // Add the assistant's response to the chat
                console.log('Adding assistant response:', data.answer);
                addMessage(data.answer, false); // Explicitly mark as assistant message

                // Log metadata for debugging (optional)
                if (data.metadata) {
                    console.log('Chat metadata:', data.metadata);
                }
            } else {
                // Add an error message to the chat
                console.error('API returned error:', data.error);
                addMessage('Sorry, I encountered an error: ' + (data.error || 'Unknown error'), false);
            }
        } catch (error) {
            // Remove loading indicator
            removeLoadingIndicator();

            // Add an error message to the chat
            addMessage('Sorry, I encountered an error: ' + error.message);
        }
    }

    // Function to clear chat history
    function clearChatHistory() {
        if (sessionId && confirm('Are you sure you want to clear the chat history?')) {
            fetch(`/api/chat/clear/${sessionId}`, {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Clear the chat messages display
                    chatMessages.innerHTML = '';

                    // Add welcome message using the addMessage function for consistency
                    addMessage("Hello! I'm the openBIS Assistant. How can I help you today?");

                    // Clear session from localStorage
                    localStorage.removeItem('chatSessionId');
                    sessionId = null;
                } else {
                    console.error('Failed to clear chat history:', data.error);
                }
            })
            .catch(error => {
                console.error('Error clearing chat history:', error);
            });
        }
    }

    // Handle form submission
    chatForm.addEventListener('submit', function(event) {
        event.preventDefault();

        const message = userInput.value.trim();
        if (message) {
            // Add the user's message to the chat
            addMessage(message, true);

            // Clear the input field
            userInput.value = '';

            // Send the message to the server
            sendMessage(message);
        }
    });

    // Handle clear chat button
    const clearChatBtn = document.getElementById('clear-chat-btn');
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', clearChatHistory);
    }

    // Focus the input field when the page loads
    userInput.focus();
});
