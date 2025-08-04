"""
Unit tests for the main module.
"""

from unittest.mock import patch, MagicMock

from chatBIS.__main__ import main


class TestMain:
    """Tests for the main module."""

    def test_main_no_command(self):
        """Test the main function with no command."""
        with patch("chatBIS.__main__.auto_mode") as mock_auto_mode:
            # Mock the auto_mode function to return 0
            mock_auto_mode.return_value = 0

            with patch("sys.argv", ["chatbis"]):
                # Call the main function
                result = main()

                # Check that auto_mode was called
                mock_auto_mode.assert_called_once()

                # Check that the function returned 0 (success)
                assert result == 0

    def test_main_invalid_command(self):
        """Test the main function with an invalid command."""
        with patch("sys.argv", ["openbis-chatbot", "invalid"]):
            with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
                with patch("argparse.ArgumentParser.print_help") as mock_print_help:
                    # Mock the argument parser to return a namespace with command="invalid"
                    mock_args = MagicMock()
                    mock_args.command = "invalid"
                    mock_parse_args.return_value = mock_args

                    # Call the main function
                    result = main()

                    # Check that the help was printed
                    mock_print_help.assert_called_once()

                    # Check that the function returned 1 (error)
                    assert result == 1
