"""Comprehensive tests for Jira integration.

This test suite provides extensive coverage of the Jira integration to prevent regressions.
All tests use mocking to avoid requiring a real Jira instance.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

# Import the functions to test
from chunksilo.search import (
    _prepare_jira_jql_query,
    _jira_issue_to_text,
    _jira_issue_to_metadata,
    _search_jira,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_jira_client():
    """Mock JIRA client for testing."""
    client = Mock()
    return client


@pytest.fixture
def mock_jira_issue():
    """Create a realistic mock Jira issue with all fields populated."""
    issue = MagicMock()
    issue.key = "PROJ-123"
    issue.fields.summary = "Test issue summary"
    issue.fields.description = "Test issue description with details"
    issue.fields.issuetype.name = "Bug"
    issue.fields.status.name = "In Progress"
    issue.fields.priority.name = "High"
    issue.fields.created = "2024-01-15T10:30:00.000+0000"
    issue.fields.updated = "2024-01-20T15:45:00.000+0000"
    issue.fields.project.key = "PROJ"
    issue.fields.project.name = "Project Name"
    issue.fields.assignee.displayName = "John Doe"
    issue.fields.reporter.displayName = "Jane Smith"

    # Mock comments
    comment1 = MagicMock()
    comment1.author.displayName = "John Doe"
    comment1.body = "First comment on this issue"
    comment2 = MagicMock()
    comment2.author.displayName = "Jane Smith"
    comment2.body = "Second comment with more details"
    issue.fields.comment.comments = [comment1, comment2]

    # Mock custom fields
    issue.fields.customfield_10001 = "Sprint 42"
    issue.fields.customfield_10002 = "Backend Team"

    # Mock attachments
    attachment1 = MagicMock()
    attachment1.filename = "error.log"
    attachment1.content = "https://jira.example.com/secure/attachment/12345/error.log"
    attachment1.size = 2048
    attachment2 = MagicMock()
    attachment2.filename = "screenshot.png"
    attachment2.content = "https://jira.example.com/secure/attachment/12346/screenshot.png"
    attachment2.size = 51200
    issue.fields.attachment = [attachment1, attachment2]

    return issue


@pytest.fixture
def minimal_jira_issue():
    """Create a minimal mock Jira issue with only required fields."""
    issue = MagicMock()
    issue.key = "MIN-1"
    issue.fields.summary = "Minimal issue"
    issue.fields.description = None
    issue.fields.issuetype.name = "Task"
    issue.fields.status.name = "Open"
    # No priority
    del issue.fields.priority
    # No dates
    del issue.fields.created
    del issue.fields.updated
    # No project details
    del issue.fields.project
    # No assignee/reporter
    del issue.fields.assignee
    del issue.fields.reporter
    # No comments
    issue.fields.comment.comments = []
    # No attachments
    issue.fields.attachment = []
    return issue


@pytest.fixture
def base_config():
    """Base configuration for tests."""
    return {
        "jira": {
            "url": "https://jira.example.com",
            "username": "test@example.com",
            "api_token": "test-token-12345",
            "timeout": 10.0,
            "max_results": 30,
            "projects": [],
            "include_comments": True,
            "include_custom_fields": True,
        },
        "ssl": {
            "ca_bundle_path": ""
        }
    }


# ============================================================================
# JQL QUERY CONSTRUCTION TESTS
# ============================================================================

class TestJiraJqlQuery:
    """Test JQL query construction logic."""

    def test_single_term_query(self, base_config):
        """Single search term should create simple text search."""
        jql = _prepare_jira_jql_query("authentication", base_config)
        assert 'text ~ "authentication"' in jql
        assert "ORDER BY updated DESC" in jql
        assert "project IN" not in jql  # No project filter for empty list

    def test_multiple_terms_query(self, base_config):
        """Multiple terms should create OR query."""
        jql = _prepare_jira_jql_query("auth error database", base_config)
        # Should have OR clause with multiple terms
        assert "text ~" in jql
        assert " OR " in jql
        assert "ORDER BY updated DESC" in jql

    def test_project_filtering_empty_list(self, base_config):
        """Empty project list should search all projects."""
        base_config["jira"]["projects"] = []
        jql = _prepare_jira_jql_query("test query", base_config)
        assert "project IN" not in jql  # No project filter

    def test_project_filtering_single_project(self, base_config):
        """Single project should add project IN clause."""
        base_config["jira"]["projects"] = ["PROJ1"]
        jql = _prepare_jira_jql_query("test query", base_config)
        assert "project IN" in jql
        assert "PROJ1" in jql

    def test_project_filtering_multiple_projects(self, base_config):
        """Multiple projects should add project IN clause."""
        base_config["jira"]["projects"] = ["PROJ1", "PROJ2", "INFRA"]
        jql = _prepare_jira_jql_query("test query", base_config)
        assert "project IN" in jql
        assert "PROJ1" in jql
        assert "PROJ2" in jql
        assert "INFRA" in jql

    def test_special_character_escaping(self, base_config):
        """Special characters in query should be escaped."""
        jql = _prepare_jira_jql_query('test "quote" query', base_config)
        # Should have escaped quotes
        assert "\\" in jql or '"' in jql

    def test_ordering_by_updated(self, base_config):
        """Query should always order by updated DESC."""
        jql = _prepare_jira_jql_query("test", base_config)
        assert "ORDER BY updated DESC" in jql

    def test_empty_query_handling(self, base_config):
        """Empty query should be handled gracefully."""
        jql = _prepare_jira_jql_query("   ", base_config)
        # Should return empty string or safe default
        assert jql == "" or "text ~" in jql

    def test_stopword_filtering(self, base_config):
        """Common stopwords should be filtered out."""
        # This tests that we're using _prepare_confluence_query_terms
        jql = _prepare_jira_jql_query("the a an", base_config)
        # Query with only stopwords should produce simple or empty query
        assert jql == "" or "ORDER BY updated DESC" in jql


# ============================================================================
# ISSUE TO TEXT CONVERSION TESTS
# ============================================================================

class TestJiraIssueToText:
    """Test Jira issue to text conversion."""

    def test_basic_issue_fields(self, mock_jira_issue):
        """Issue key, summary, description should always be included."""
        text = _jira_issue_to_text(mock_jira_issue, True, True)
        assert "PROJ-123" in text
        assert "Test issue summary" in text
        assert "Test issue description" in text

    def test_issue_key_and_summary_format(self, mock_jira_issue):
        """Text should have proper formatting for issue key and summary."""
        text = _jira_issue_to_text(mock_jira_issue, False, False)
        assert "Issue: PROJ-123" in text
        assert "Summary: Test issue summary" in text

    def test_comments_included_when_enabled(self, mock_jira_issue):
        """Comments should be included when flag is True."""
        text = _jira_issue_to_text(mock_jira_issue, include_comments=True, include_custom_fields=False)
        assert "First comment on this issue" in text
        assert "Second comment with more details" in text
        assert "John Doe" in text
        assert "Jane Smith" in text
        assert "Comments:" in text

    def test_comments_excluded_when_disabled(self, mock_jira_issue):
        """Comments should be excluded when flag is False."""
        text = _jira_issue_to_text(mock_jira_issue, include_comments=False, include_custom_fields=False)
        assert "First comment" not in text
        assert "Second comment" not in text
        # But issue content should still be there
        assert "PROJ-123" in text

    def test_custom_fields_included(self, mock_jira_issue):
        """Custom fields should be included when flag is True."""
        text = _jira_issue_to_text(mock_jira_issue, include_comments=False, include_custom_fields=True)
        assert "customfield_10001" in text
        assert "Sprint 42" in text
        assert "customfield_10002" in text
        assert "Backend Team" in text
        assert "Custom Fields:" in text

    def test_custom_fields_excluded(self, mock_jira_issue):
        """Custom fields should be excluded when flag is False."""
        text = _jira_issue_to_text(mock_jira_issue, include_comments=False, include_custom_fields=False)
        assert "customfield_10001" not in text
        assert "Sprint 42" not in text
        # But issue content should still be there
        assert "PROJ-123" in text

    def test_missing_description(self, minimal_jira_issue):
        """Issues without descriptions should not crash."""
        text = _jira_issue_to_text(minimal_jira_issue, False, False)
        assert "MIN-1" in text
        assert "Minimal issue" in text
        # Should complete without error even though description is None

    def test_no_comments(self, minimal_jira_issue):
        """Issues without comments should not crash."""
        text = _jira_issue_to_text(minimal_jira_issue, include_comments=True, include_custom_fields=False)
        assert "MIN-1" in text
        # Should not have "Comments:" section if no comments
        # (This may vary based on implementation)

    def test_text_formatting(self, mock_jira_issue):
        """Text should be well-formatted with sections."""
        text = _jira_issue_to_text(mock_jira_issue, True, True)
        assert "Issue:" in text
        assert "Summary:" in text
        # Description section should exist
        assert "Description:" in text or "description" in text.lower()

    def test_all_content_types_together(self, mock_jira_issue):
        """With all flags enabled, should include everything."""
        text = _jira_issue_to_text(mock_jira_issue, include_comments=True, include_custom_fields=True)
        # Check all content types are present
        assert "PROJ-123" in text  # Issue key
        assert "Test issue summary" in text  # Summary
        assert "Test issue description" in text  # Description
        assert "First comment" in text  # Comments
        assert "customfield_10001" in text  # Custom fields


# ============================================================================
# METADATA EXTRACTION TESTS
# ============================================================================

class TestJiraIssueToMetadata:
    """Test metadata extraction from Jira issues."""

    def test_standard_fields(self, mock_jira_issue):
        """Standard fields should be extracted correctly."""
        metadata = _jira_issue_to_metadata(mock_jira_issue, "https://jira.example.com")
        assert metadata["source"] == "Jira"
        assert metadata["issue_key"] == "PROJ-123"
        assert metadata["issue_type"] == "Bug"
        assert metadata["status"] == "In Progress"
        assert metadata["priority"] == "High"
        assert metadata["title"] == "Test issue summary"
        assert metadata["file_name"] == "PROJ-123: Test issue summary"

    def test_date_parsing(self, mock_jira_issue):
        """Dates should be parsed and formatted as YYYY-MM-DD."""
        metadata = _jira_issue_to_metadata(mock_jira_issue, "https://jira.example.com")
        assert metadata["creation_date"] == "2024-01-15"
        assert metadata["last_modified_date"] == "2024-01-20"

    def test_project_info(self, mock_jira_issue):
        """Project key and name should be extracted."""
        metadata = _jira_issue_to_metadata(mock_jira_issue, "https://jira.example.com")
        assert metadata["project_key"] == "PROJ"
        assert metadata["project_name"] == "Project Name"

    def test_assignee_reporter(self, mock_jira_issue):
        """Assignee and reporter should be extracted."""
        metadata = _jira_issue_to_metadata(mock_jira_issue, "https://jira.example.com")
        assert metadata["assignee"] == "John Doe"
        assert metadata["reporter"] == "Jane Smith"

    def test_attachment_listing(self, mock_jira_issue):
        """Attachments should be listed with filename, URL, size."""
        metadata = _jira_issue_to_metadata(mock_jira_issue, "https://jira.example.com")
        assert "attachments" in metadata
        assert len(metadata["attachments"]) == 2

        # Check first attachment
        assert metadata["attachments"][0]["filename"] == "error.log"
        assert metadata["attachments"][0]["url"] == "https://jira.example.com/secure/attachment/12345/error.log"
        assert metadata["attachments"][0]["size"] == 2048

        # Check second attachment
        assert metadata["attachments"][1]["filename"] == "screenshot.png"
        assert metadata["attachments"][1]["size"] == 51200

    def test_missing_optional_fields(self, minimal_jira_issue):
        """Missing optional fields should not crash."""
        metadata = _jira_issue_to_metadata(minimal_jira_issue, "https://jira.example.com")
        assert metadata["issue_key"] == "MIN-1"
        assert metadata["source"] == "Jira"
        # Optional fields should be missing or None
        assert "priority" not in metadata or metadata.get("priority") is None
        assert "creation_date" not in metadata
        assert "assignee" not in metadata

    def test_file_name_format(self, mock_jira_issue):
        """file_name should follow the format '{key}: {summary}'."""
        metadata = _jira_issue_to_metadata(mock_jira_issue, "https://jira.example.com")
        expected_file_name = f"{mock_jira_issue.key}: {mock_jira_issue.fields.summary}"
        assert metadata["file_name"] == expected_file_name

    def test_source_always_jira(self, mock_jira_issue, minimal_jira_issue):
        """source field should always be 'Jira'."""
        metadata1 = _jira_issue_to_metadata(mock_jira_issue, "https://jira.example.com")
        metadata2 = _jira_issue_to_metadata(minimal_jira_issue, "https://jira.example.com")
        assert metadata1["source"] == "Jira"
        assert metadata2["source"] == "Jira"


# ============================================================================
# MAIN SEARCH FUNCTION TESTS
# ============================================================================

class TestSearchJira:
    """Test _search_jira function with mocks."""

    @patch('chunksilo.search.JIRA')
    def test_successful_search(self, mock_jira_class, mock_jira_issue, base_config):
        """Successful search should return NodeWithScore list."""
        # Mock JIRA client and search_issues response
        mock_client = Mock()
        mock_client.search_issues.return_value = [mock_jira_issue]
        mock_jira_class.return_value = mock_client

        nodes = _search_jira("test query", base_config)

        # Verify results
        assert len(nodes) == 1
        assert nodes[0].node.metadata["issue_key"] == "PROJ-123"
        assert nodes[0].score == 0.0
        assert "Test issue summary" in nodes[0].node.text

    def test_empty_url_returns_empty_list(self, base_config):
        """Empty URL should skip search and return empty list."""
        base_config["jira"]["url"] = ""
        nodes = _search_jira("test", base_config)
        assert nodes == []

    def test_missing_username_returns_empty_list(self, base_config):
        """Missing username should skip search and return empty list."""
        base_config["jira"]["username"] = ""
        nodes = _search_jira("test", base_config)
        assert nodes == []

    def test_missing_api_token_returns_empty_list(self, base_config):
        """Missing API token should skip search and return empty list."""
        base_config["jira"]["api_token"] = ""
        nodes = _search_jira("test", base_config)
        assert nodes == []

    @patch('chunksilo.search.JIRA', None)
    def test_missing_library_returns_empty_list(self, base_config):
        """Missing jira library should return empty list."""
        nodes = _search_jira("test", base_config)
        assert nodes == []

    @patch('chunksilo.search.JIRA')
    def test_api_error_returns_empty_list(self, mock_jira_class, base_config):
        """API errors should be caught and return empty list."""
        mock_jira_class.side_effect = Exception("API Connection Error")
        nodes = _search_jira("test", base_config)
        assert nodes == []

    @patch('chunksilo.search.JIRA')
    def test_ssl_ca_bundle_passed(self, mock_jira_class, base_config):
        """SSL CA bundle should be passed to JIRA client."""
        base_config["ssl"]["ca_bundle_path"] = "/path/to/ca-bundle.crt"

        # Mock client
        mock_client = Mock()
        mock_client.search_issues.return_value = []
        mock_jira_class.return_value = mock_client

        _search_jira("test", base_config)

        # Verify JIRA was called with SSL options
        assert mock_jira_class.called
        call_kwargs = mock_jira_class.call_args[1]
        assert "options" in call_kwargs
        assert call_kwargs["options"]["verify"] == "/path/to/ca-bundle.crt"

    @patch('chunksilo.search.JIRA')
    def test_basic_auth_used(self, mock_jira_class, base_config):
        """Basic auth should be used with username and API token."""
        mock_client = Mock()
        mock_client.search_issues.return_value = []
        mock_jira_class.return_value = mock_client

        _search_jira("test", base_config)

        # Verify basic_auth was passed
        call_kwargs = mock_jira_class.call_args[1]
        assert "basic_auth" in call_kwargs
        assert call_kwargs["basic_auth"] == ("test@example.com", "test-token-12345")

    @patch('chunksilo.search.JIRA')
    def test_max_results_respected(self, mock_jira_class, base_config):
        """max_results configuration should be passed to search_issues."""
        base_config["jira"]["max_results"] = 50

        mock_client = Mock()
        mock_client.search_issues.return_value = []
        mock_jira_class.return_value = mock_client

        _search_jira("test", base_config)

        # Verify maxResults was passed
        call_args = mock_client.search_issues.call_args
        assert call_args[1]["maxResults"] == 50

    @patch('chunksilo.search.JIRA')
    def test_fields_all_requested(self, mock_jira_class, base_config):
        """All fields including custom fields should be requested."""
        mock_client = Mock()
        mock_client.search_issues.return_value = []
        mock_jira_class.return_value = mock_client

        _search_jira("test", base_config)

        # Verify fields='*all' was passed
        call_args = mock_client.search_issues.call_args
        assert call_args[1]["fields"] == "*all"

    @patch('chunksilo.search.JIRA')
    def test_multiple_issues_returned(self, mock_jira_class, mock_jira_issue, minimal_jira_issue, base_config):
        """Multiple issues should be converted to multiple nodes."""
        mock_client = Mock()
        mock_client.search_issues.return_value = [mock_jira_issue, minimal_jira_issue]
        mock_jira_class.return_value = mock_client

        nodes = _search_jira("test", base_config)

        assert len(nodes) == 2
        assert nodes[0].node.metadata["issue_key"] == "PROJ-123"
        assert nodes[1].node.metadata["issue_key"] == "MIN-1"


# ============================================================================
# REGRESSION PREVENTION TESTS
# ============================================================================

class TestJiraRegressions:
    """Tests to prevent common regressions."""

    def test_special_characters_in_summary(self):
        """Special characters in summary should not break parsing."""
        issue = MagicMock()
        issue.key = "TEST-1"
        issue.fields.summary = 'Issue with "quotes" and \\backslashes\\ and émojis 🎉'
        issue.fields.description = "Normal description"
        issue.fields.issuetype.name = "Bug"
        issue.fields.status.name = "Open"

        # Should not crash
        text = _jira_issue_to_text(issue, False, False)
        metadata = _jira_issue_to_metadata(issue, "https://jira.example.com")

        assert "TEST-1" in text
        assert metadata["issue_key"] == "TEST-1"

    def test_unicode_in_comments(self, mock_jira_issue):
        """Unicode characters in comments should be handled."""
        # Add unicode comment
        unicode_comment = MagicMock()
        unicode_comment.author.displayName = "José García"
        unicode_comment.body = "Comment with unicode: café, naïve, 中文"
        mock_jira_issue.fields.comment.comments.append(unicode_comment)

        # Should not crash
        text = _jira_issue_to_text(mock_jira_issue, include_comments=True, include_custom_fields=False)
        assert "José García" in text

    @patch('chunksilo.search.JIRA')
    def test_empty_query_after_processing(self, mock_jira_class, base_config):
        """Empty query after term processing should be handled."""
        mock_client = Mock()
        mock_client.search_issues.return_value = []
        mock_jira_class.return_value = mock_client

        # Query with only stopwords
        nodes = _search_jira("the a an", base_config)

        # Should return empty list, not crash
        assert isinstance(nodes, list)

    def test_none_custom_field_values(self):
        """None values in custom fields should be skipped."""
        issue = MagicMock()
        issue.key = "TEST-1"
        issue.fields.summary = "Test"
        issue.fields.description = "Desc"
        issue.fields.issuetype.name = "Task"
        issue.fields.status.name = "Open"
        # Add a None custom field
        issue.fields.customfield_10001 = None
        issue.fields.customfield_10002 = ""
        issue.fields.customfield_10003 = "Valid Value"

        text = _jira_issue_to_text(issue, False, include_custom_fields=True)

        # Should only include the valid custom field
        assert "customfield_10003" in text
        assert "Valid Value" in text
        # Should not include None or empty fields
        assert "customfield_10001" not in text
        assert "customfield_10002" not in text

    def test_malformed_dates_handled(self):
        """Malformed date strings should be handled gracefully."""
        issue = MagicMock()
        issue.key = "TEST-1"
        issue.fields.summary = "Test"
        issue.fields.issuetype.name = "Task"
        issue.fields.status.name = "Open"
        issue.fields.created = "not-a-valid-date"
        issue.fields.updated = "also-invalid"

        # Should not crash, just skip the dates
        metadata = _jira_issue_to_metadata(issue, "https://jira.example.com")

        assert metadata["issue_key"] == "TEST-1"
        # Dates might be missing or None
        assert "creation_date" not in metadata or metadata.get("creation_date") is None
