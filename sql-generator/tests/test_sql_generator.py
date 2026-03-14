"""Tests for sql-generator: schema_loader, executor, evaluator, generator."""
from __future__ import annotations

import pathlib
import sqlite3
import sys

import pytest

SRC = pathlib.Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

CHINOOK_PATH = pathlib.Path(__file__).parent.parent / "data" / "chinook" / "Chinook.sqlite"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def chinook_db():
    if not CHINOOK_PATH.exists():
        pytest.skip("Chinook.sqlite not downloaded yet")
    return CHINOOK_PATH


@pytest.fixture(scope="session")
def tiny_db(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "tiny.sqlite"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE Artist (ArtistId INTEGER PRIMARY KEY, Name TEXT)")
    cur.execute("INSERT INTO Artist VALUES (1, 'AC/DC')")
    cur.execute("INSERT INTO Artist VALUES (2, 'Accept')")
    cur.execute("CREATE TABLE Album (AlbumId INTEGER PRIMARY KEY, Title TEXT, ArtistId INTEGER)")
    cur.execute("INSERT INTO Album VALUES (1, 'For Those About To Rock', 1)")
    cur.execute("INSERT INTO Album VALUES (2, 'Balls to the Wall', 2)")
    cur.execute("INSERT INTO Album VALUES (3, 'Restless and Wild', 2)")
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# TestLoadSchema
# ---------------------------------------------------------------------------

class TestLoadSchema:
    def test_returns_dict(self, chinook_db):
        from schema_loader import load_schema
        result = load_schema(chinook_db)
        assert isinstance(result, dict)

    def test_has_expected_tables(self, chinook_db):
        from schema_loader import load_schema
        result = load_schema(chinook_db)
        expected = {"Album", "Artist", "Track", "Invoice", "Customer"}
        assert expected.issubset(result.keys())

    def test_table_entry_has_required_keys(self, chinook_db):
        from schema_loader import load_schema
        result = load_schema(chinook_db)
        for meta in result.values():
            assert "columns" in meta
            assert "sample_rows" in meta

    def test_columns_is_list_of_dicts(self, chinook_db):
        from schema_loader import load_schema
        result = load_schema(chinook_db)
        for meta in result.values():
            assert isinstance(meta["columns"], list)
            for col in meta["columns"]:
                assert "name" in col
                assert "type" in col

    def test_sample_rows_max_three(self, chinook_db):
        from schema_loader import load_schema
        result = load_schema(chinook_db)
        for meta in result.values():
            assert len(meta["sample_rows"]) <= 3

    def test_sample_rows_are_lists(self, chinook_db):
        from schema_loader import load_schema
        result = load_schema(chinook_db)
        for meta in result.values():
            for row in meta["sample_rows"]:
                assert isinstance(row, (list, tuple))

    def test_works_with_string_path(self, chinook_db):
        from schema_loader import load_schema
        result = load_schema(str(chinook_db))
        assert isinstance(result, dict)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# TestFormatSchemaContext
# ---------------------------------------------------------------------------

class TestFormatSchemaContext:
    def test_returns_string(self, chinook_db):
        from schema_loader import load_schema, format_schema_context
        schema = load_schema(chinook_db)
        result = format_schema_context(schema)
        assert isinstance(result, str)

    def test_contains_table_names(self, chinook_db):
        from schema_loader import load_schema, format_schema_context
        schema = load_schema(chinook_db)
        result = format_schema_context(schema)
        for table in schema.keys():
            assert table in result

    def test_contains_column_names(self, chinook_db):
        from schema_loader import load_schema, format_schema_context
        schema = load_schema(chinook_db)
        result = format_schema_context(schema)
        assert "ArtistId" in result
        assert "Name" in result

    def test_non_empty(self, chinook_db):
        from schema_loader import load_schema, format_schema_context
        schema = load_schema(chinook_db)
        result = format_schema_context(schema)
        assert len(result) > 0

    def test_minimal_schema_dict(self):
        from schema_loader import format_schema_context
        minimal = {
            "Foo": {
                "columns": [{"name": "id", "type": "INTEGER"}, {"name": "val", "type": "TEXT"}],
                "sample_rows": [[1, "hello"]],
            }
        }
        result = format_schema_context(minimal)
        assert "Foo" in result
        assert "id" in result
        assert "val" in result


# ---------------------------------------------------------------------------
# TestExecuteQuery
# ---------------------------------------------------------------------------

class TestExecuteQuery:
    def test_returns_dataframe(self, tiny_db):
        import pandas as pd
        from executor import execute_query
        df = execute_query("SELECT * FROM Artist", tiny_db)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, tiny_db):
        from executor import execute_query
        df = execute_query("SELECT * FROM Artist", tiny_db)
        assert len(df) == 2

    def test_correct_columns(self, tiny_db):
        from executor import execute_query
        df = execute_query("SELECT ArtistId, Name FROM Artist", tiny_db)
        assert list(df.columns) == ["ArtistId", "Name"]

    def test_filtered_query(self, tiny_db):
        from executor import execute_query
        df = execute_query("SELECT * FROM Album WHERE ArtistId = 2", tiny_db)
        assert len(df) == 2

    def test_accepts_string_path(self, tiny_db):
        from executor import execute_query
        df = execute_query("SELECT * FROM Artist", str(tiny_db))
        assert len(df) == 2

    def test_accepts_pathlib_path(self, tiny_db):
        import pathlib
        from executor import execute_query
        df = execute_query("SELECT * FROM Artist", pathlib.Path(tiny_db))
        assert len(df) == 2

    def test_bad_sql_raises_value_error(self, tiny_db):
        from executor import execute_query
        with pytest.raises(ValueError):
            execute_query("SELECT * FROM NonExistentTable", tiny_db)

    def test_empty_result_returns_empty_dataframe(self, tiny_db):
        import pandas as pd
        from executor import execute_query
        df = execute_query("SELECT * FROM Artist WHERE ArtistId = 9999", tiny_db)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_join_query(self, tiny_db):
        from executor import execute_query
        sql = "SELECT Artist.Name, Album.Title FROM Artist JOIN Album ON Artist.ArtistId = Album.ArtistId"
        df = execute_query(sql, tiny_db)
        assert len(df) == 3
        assert "Name" in df.columns
        assert "Title" in df.columns


# ---------------------------------------------------------------------------
# TestEvaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_dict(self, tiny_db):
        from evaluator import evaluate
        result = evaluate("SELECT * FROM Artist", "SELECT * FROM Artist", tiny_db)
        assert isinstance(result, dict)

    def test_required_keys(self, tiny_db):
        from evaluator import evaluate
        result = evaluate("SELECT * FROM Artist", "SELECT * FROM Artist", tiny_db)
        assert {"match", "generated_result_shape", "reference_result_shape", "results_match"}.issubset(result.keys())

    def test_exact_match_true(self, tiny_db):
        from evaluator import evaluate
        result = evaluate("SELECT * FROM Artist", "SELECT * FROM Artist", tiny_db)
        assert result["match"] is True

    def test_exact_match_case_insensitive(self, tiny_db):
        from evaluator import evaluate
        result = evaluate("select * from artist", "SELECT * FROM Artist", tiny_db)
        assert result["match"] is True

    def test_exact_match_strips_whitespace(self, tiny_db):
        from evaluator import evaluate
        result = evaluate("  SELECT * FROM Artist  ", "SELECT * FROM Artist", tiny_db)
        assert result["match"] is True

    def test_different_sql_match_false(self, tiny_db):
        from evaluator import evaluate
        result = evaluate("SELECT * FROM Artist", "SELECT * FROM Album", tiny_db)
        assert result["match"] is False

    def test_results_match_true_same_data(self, tiny_db):
        from evaluator import evaluate
        result = evaluate("SELECT * FROM Artist", "SELECT * FROM Artist", tiny_db)
        assert result["results_match"] is True

    def test_results_match_false_different_data(self, tiny_db):
        from evaluator import evaluate
        result = evaluate("SELECT * FROM Artist", "SELECT * FROM Album", tiny_db)
        assert result["results_match"] is False

    def test_shape_fields_are_tuples(self, tiny_db):
        from evaluator import evaluate
        result = evaluate("SELECT * FROM Artist", "SELECT * FROM Artist", tiny_db)
        assert isinstance(result["generated_result_shape"], tuple)
        assert isinstance(result["reference_result_shape"], tuple)

    def test_shape_correct_dimensions(self, tiny_db):
        from evaluator import evaluate
        result = evaluate("SELECT * FROM Artist", "SELECT * FROM Artist", tiny_db)
        assert result["generated_result_shape"] == (2, 2)


# ---------------------------------------------------------------------------
# TestGeneratorMocked
# ---------------------------------------------------------------------------

class TestGeneratorMocked:
    def test_generate_sql_calls_anthropic(self, monkeypatch):
        try:
            import generator
        except ImportError:
            pytest.skip("generator.py not implemented yet")

        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="SELECT * FROM Artist")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic_cls = MagicMock(return_value=mock_client)

        with patch.object(generator.anthropic, "Anthropic", mock_anthropic_cls):
            result = generator.generate_sql("List all artists", "TABLE Artist\n  Columns: ArtistId (INTEGER), Name (TEXT)\n")

        mock_client.messages.create.assert_called_once()
        assert "SELECT" in result

    def test_generate_sql_strips_markdown_fences(self, monkeypatch):
        try:
            import generator
        except ImportError:
            pytest.skip("generator.py not implemented yet")

        from unittest.mock import MagicMock, patch

        sql_body = "SELECT * FROM Artist"
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=f"```sql\n{sql_body}\n```")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic_cls = MagicMock(return_value=mock_client)

        with patch.object(generator.anthropic, "Anthropic", mock_anthropic_cls):
            result = generator.generate_sql("List all artists", "schema here")

        assert "```" not in result
        assert result == sql_body

    def test_generate_sql_reads_api_key_from_env(self, monkeypatch):
        try:
            import generator
        except ImportError:
            pytest.skip("generator.py not implemented yet")

        from unittest.mock import MagicMock, patch
        import inspect

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc123")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="SELECT 1")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic_cls = MagicMock(return_value=mock_client)

        with patch.object(generator.anthropic, "Anthropic", mock_anthropic_cls):
            generator.generate_sql("test", "schema")

        # Verify no hardcoded key in source
        source = inspect.getsource(generator)
        assert "sk-ant-" not in source
        assert "AKIA" not in source
