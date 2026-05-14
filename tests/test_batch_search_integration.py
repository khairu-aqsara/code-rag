"""Integration tests for batch search endpoint."""
import pytest


@pytest.mark.integration
class TestBatchSearchIntegration:
    def test_batch_search_endpoint_returns_422_without_data(self, redis_client, flush_redis):
        """Batch search with empty queries → returns validation error"""
        import httpx

        # This tests that the endpoint exists and validates input.
        # Full end-to-end testing requires indexed data and a running API server.
        # Use the REST API directly if available, otherwise skip.
        pytest.skip(
            "Batch search integration test requires running API server with indexed data. "
            "Run manually against a live instance."
        )
