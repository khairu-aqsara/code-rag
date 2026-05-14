"""Unit tests for MCP error handling with actionable suggestions."""
import importlib.util
import pytest
from unittest.mock import MagicMock
import httpx


def _load_handle_error():
    """Load handle_error from mcp/server.py via file path (avoids mcp package conflict)."""
    import os
    server_path = os.path.join(os.path.dirname(__file__), "..", "mcp", "server.py")
    server_path = os.path.abspath(server_path)
    spec = importlib.util.spec_from_file_location("mcp_server_local", server_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.handle_error


handle_error = _load_handle_error()


@pytest.mark.unit
class TestMCPErrorHandling:
    def test_project_not_found_suggests_list_projects(self):
        """404 error → suggests calling list_projects"""
        response_mock = MagicMock()
        response_mock.status_code = 404
        response_mock.json.return_value = {"detail": "Project not found"}

        e = httpx.HTTPStatusError(
            "Not found",
            request=MagicMock(),
            response=response_mock
        )

        result = handle_error(e)

        assert "coderag_list_projects" in result
        assert "suggestion" in result.lower() or "Suggestion" in result

    def test_empty_results_suggests_alternatives(self):
        """Empty results → suggests exact search and filter adjustments"""
        e = ValueError("No results found for query")

        result = handle_error(e)

        assert "coderag_search_code_exact" in result
        assert "min_score" in result

    def test_connection_error_suggests_checking_app(self):
        """Connection error → suggests checking if app is running"""
        e = httpx.ConnectError("Connection refused")

        result = handle_error(e)

        assert "app" in result.lower() or "APP_URL" in result

    def test_timeout_suggests_waiting_for_models(self):
        """Timeout → suggests waiting for model loading"""
        e = httpx.TimeoutException("Request timed out")

        result = handle_error(e)

        assert "model" in result.lower() or "wait" in result.lower()
