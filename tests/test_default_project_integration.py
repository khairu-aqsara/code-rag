"""Integration tests for default project configuration endpoint."""
import pytest


@pytest.mark.integration
class TestDefaultProjectIntegration:
    def test_set_and_get_round_trip(self, redis_client):
        """Set project → get returns same value"""
        redis_client.set("config:default_project:test-workspace", "test-project")
        
        result = redis_client.get("config:default_project:test-workspace")
        
        assert result == b"test-project"
    
    def test_overwrite_default_project(self, redis_client):
        """Setting new project_id overwrites existing"""
        redis_client.set("config:default_project:workspace", "project-1")
        redis_client.set("config:default_project:workspace", "project-2")
        
        result = redis_client.get("config:default_project:workspace")
        
        assert result == b"project-2"
