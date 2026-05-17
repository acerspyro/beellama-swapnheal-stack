import runpy
import sys
import pytest
from unittest.mock import patch, MagicMock

def test_main_execution():
    # Mocking sys.argv to simulate command line arguments
    test_args = ["prog", "--upstream", "http://test-upstream", "--port", "1234", "--timeout", "300.0", "--host", "0.0.0.0"]
    
    # We need to patch build_app and uvicorn before running the module
    with patch("sys.argv", test_args), \
         patch("uvicorn.run") as mock_run, \
         patch("tool_healing.api.build_app") as mock_build:
        
        mock_app = MagicMock()
        mock_build.return_value = mock_app
        
        # Execute the __main__.py file
        # Use absolute path to be safe
        import os
        main_path = os.path.join(os.path.dirname(__file__), "..", "__main__.py")
        runpy.run_path(main_path, run_name="__main__")
        
        # Verify uvicorn.run was called with expected arguments
        mock_run.assert_called_once_with(mock_app, host="0.0.0.0", port=1234)
        
        # Verify build_app was called with the upstream from args
        mock_build.assert_any_call("http://test-upstream", default_timeout=300.0)

def test_main_defaults():
    test_args = ["prog"]
    with patch("sys.argv", test_args), \
         patch("uvicorn.run") as mock_run, \
         patch("tool_healing.api.build_app") as mock_build:
        
        mock_app = MagicMock()
        mock_build.return_value = mock_app
        
        import os
        main_path = os.path.join(os.path.dirname(__file__), "..", "__main__.py")
        runpy.run_path(main_path, run_name="__main__")
        
        mock_run.assert_called_once_with(mock_app, host="127.0.0.1", port=8081)
        mock_build.assert_any_call("http://127.0.0.1:8080", default_timeout=600.0)
