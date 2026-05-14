"""Integration tests for staleness detection — last_indexed timestamp in Redis."""
import pytest


@pytest.mark.integration
class TestStalenessIntegration:
    def test_last_indexed_stored_after_set(self, redis_client, flush_redis):
        """Setting last_indexed key → value is retrievable"""
        redis_client.set("meta:testproject:last_indexed", "1715334000")

        result = redis_client.get("meta:testproject:last_indexed")

        assert result == b"1715334000"

    def test_last_indexed_overwrite(self, redis_client, flush_redis):
        """Setting last_indexed twice → latest value wins"""
        redis_client.set("meta:testproject:last_indexed", "1000000000")
        redis_client.set("meta:testproject:last_indexed", "1715334000")

        result = redis_client.get("meta:testproject:last_indexed")

        assert result == b"1715334000"

    def test_last_indexed_missing_returns_none(self, redis_client, flush_redis):
        """No last_indexed key → returns None"""
        result = redis_client.get("meta:nonexistent:last_indexed")

        assert result is None
