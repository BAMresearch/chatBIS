"""
Unit tests for the query CLI.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock, call

from src.chatBIS.query.cli import main


class TestQueryCLI:
    """Tests for the query CLI."""

    @patch("src.chatBIS.query.cli.RAGQueryEngine")
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("sys.argv", ["chatbis", "--data", "./data", "--verbose"])
    def test_main_success(self, mock_print, mock_input, mock_engine_class):
        """Test the main function with valid arguments."""
        # Mock the query engine instance
        mock_engine = MagicMock()
        mock_engine.query.return_value = ("Test answer", [
            {"title": "Test 1", "url": "https://example.com/1"},
            {"title": "Test 2", "url": "https://example.com/2"}
        ])
        mock_engine_class.return_value = mock_engine

        # Mock the user input
        mock_input.side_effect = ["Test query", "exit"]

        # Call the main function
        result = main()

        # Check that the query engine was created with the correct arguments
        mock_engine_class.assert_called_once_with(
            data_dir="./data",
            api_key=None,
            model="qwen3"
        )

        # Check that the query method was called
        mock_engine.query.assert_called_once_with("Test query", top_k=5)

        # Check that the answer was printed
        mock_print.assert_any_call("\nAssistant: Test answer\n")

        # Check that the function returned 0 (success)
        assert result == 0

    @patch("src.openbis_chatbot.query.cli.RAGQueryEngine")
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("sys.argv", ["openbis-chatbot", "--data", "./data", "--api-key", "test_key",
                       "--model", "test_model", "--top-k", "5"])
    def test_main_with_all_arguments(self, mock_print, mock_input, mock_engine_class):
        """Test the main function with all arguments."""
        # Mock the query engine instance
        mock_engine = MagicMock()
        mock_engine.query.return_value = ("Test answer", [
            {"title": "Test 1", "url": "https://example.com/1"},
            {"title": "Test 2", "url": "https://example.com/2"}
        ])
        mock_engine_class.return_value = mock_engine

        # Mock the user input
        mock_input.side_effect = ["Test query", "exit"]

        # Call the main function
        result = main()

        # Check that the query engine was created with the correct arguments
        mock_engine_class.assert_called_once_with(
            data_dir="./data",
            api_key="test_key",
            model="test_model"
        )

        # Check that the query method was called
        mock_engine.query.assert_called_once_with("Test query", top_k=5)

        # Check that the answer was printed
        mock_print.assert_any_call("\nAssistant: Test answer\n")

        # Check that the function returned 0 (success)
        assert result == 0

    @patch("src.chatBIS.query.cli.RAGQueryEngine")
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("sys.argv", ["chatbis", "--data", "./data"])
    def test_main_with_empty_query(self, mock_print, mock_input, mock_engine_class):
        """Test the main function with an empty query."""
        # Mock the query engine instance
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        # Mock the user input
        mock_input.side_effect = ["", "exit"]

        # Call the main function
        result = main()

        # Check that the query method was not called
        mock_engine.query.assert_not_called()

        # Check that the function returned 0 (success)
        assert result == 0

    @patch("src.openbis_chatbot.query.cli.RAGQueryEngine")
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("sys.argv", ["openbis-chatbot", "--data", "./data"])
    def test_main_with_query_error(self, mock_print, mock_input, mock_engine_class):
        """Test the main function when a query error occurs."""
        # Mock the query engine instance
        mock_engine = MagicMock()
        mock_engine.query.side_effect = Exception("Test error")
        mock_engine_class.return_value = mock_engine

        # Mock the user input
        mock_input.side_effect = ["Test query", "exit"]

        # Call the main function
        result = main()

        # Check that the query method was called
        mock_engine.query.assert_called_once_with("Test query", top_k=5)

        # Check that the error message was printed
        mock_print.assert_any_call("I'm sorry, I encountered an error. Please try again or ask a different question.")

        # Check that the function returned 0 (success)
        assert result == 0

    @patch("src.openbis_chatbot.query.cli.RAGQueryEngine")
    @patch("sys.argv", ["openbis-chatbot", "--data", "./data"])
    def test_main_with_init_error(self, mock_engine_class):
        """Test the main function when an initialization error occurs."""
        # Mock the query engine class to raise an exception
        mock_engine_class.side_effect = Exception("Test error")

        # Call the main function
        result = main()

        # Check that the function returned 1 (error)
        assert result == 1
