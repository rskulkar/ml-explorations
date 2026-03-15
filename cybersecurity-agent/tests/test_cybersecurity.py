"""Tests for the cybersecurity-agent RAG pipeline.

All external I/O (Claude API, ChromaDB, sentence-transformers, PDF files)
is mocked so that the test suite runs offline with no API keys required.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — mirrors sql-generator test pattern
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent
SRC = _HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ingester
import vectorstore as vs
import agent
import pipeline


# ===========================================================================
# Helpers
# ===========================================================================

def _make_fake_pdf_path() -> pathlib.Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(b"%PDF-1.4 fake")
    tmp.flush()
    tmp.close()
    return pathlib.Path(tmp.name)


# ===========================================================================
# TestIngester
# ===========================================================================

class TestIngester:
    def test_load_pdf_returns_string(self):
        pdf_path = _make_fake_pdf_path()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "hello world"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        with patch.object(ingester, "PdfReader", return_value=mock_reader):
            result = ingester.load_pdf(pdf_path)
        assert isinstance(result, str)
        assert "hello world" in result

    def test_load_pdf_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ingester.load_pdf(pathlib.Path("/nonexistent/path/file.pdf"))

    def test_chunk_text_basic(self):
        text = "A" * 3000
        chunks = ingester.chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 2
        assert all(isinstance(c, str) for c in chunks)
        assert all(len(c) > 0 for c in chunks)

    def test_chunk_text_overlap(self):
        # chunk_size=500 tokens → 2000 chars; overlap=50 tokens → 200 chars; step=1800
        text = "X" * 4200
        chunks = ingester.chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 2
        # Last 200 chars of chunk[0] should equal first 200 chars of chunk[1]
        assert chunks[0][-200:] == chunks[1][:200]

    def test_chunk_text_short_text_single_chunk(self):
        text = "Short text"
        chunks = ingester.chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_chunk_text_empty_returns_empty(self):
        chunks = ingester.chunk_text("", chunk_size=500, overlap=50)
        assert chunks == []


# ===========================================================================
# TestVectorstore
# ===========================================================================

class TestVectorstore:
    def _make_mock_st(self, n_chunks: int = 3):
        import numpy as np
        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros((n_chunks, 384), dtype="float32")
        return mock_st

    def test_build_vectorstore_calls_encode(self):
        chunks = ["chunk one", "chunk two", "chunk three"]
        mock_st = self._make_mock_st(3)
        mock_col = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col
        with patch.object(vs, "SentenceTransformer", return_value=mock_st), \
             patch.object(vs, "chromadb") as mock_chroma:
            mock_chroma.Client.return_value = mock_client
            vs.build_vectorstore(chunks)
        mock_st.encode.assert_called_once_with(chunks)

    def test_build_vectorstore_adds_correct_count(self):
        chunks = ["a", "b", "c", "d"]
        mock_st = self._make_mock_st(4)
        mock_col = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col
        with patch.object(vs, "SentenceTransformer", return_value=mock_st), \
             patch.object(vs, "chromadb") as mock_chroma:
            mock_chroma.Client.return_value = mock_client
            vs.build_vectorstore(chunks)
        call_kwargs = mock_col.add.call_args
        assert len(call_kwargs.kwargs["ids"]) == 4

    def test_query_vectorstore_returns_list(self):
        import numpy as np
        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros((1, 384), dtype="float32")
        mock_col = MagicMock()
        mock_col.query.return_value = {"documents": [["relevant doc"]]}
        with patch.object(vs, "SentenceTransformer", return_value=mock_st):
            result = vs.query_vectorstore(mock_col, "some query", k=1)
        assert isinstance(result, list)
        assert result == ["relevant doc"]

    def test_query_vectorstore_k_respected(self):
        import numpy as np
        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros((1, 384), dtype="float32")
        mock_col = MagicMock()
        mock_col.query.return_value = {"documents": [["d1", "d2", "d3"]]}
        with patch.object(vs, "SentenceTransformer", return_value=mock_st):
            vs.query_vectorstore(mock_col, "query", k=3)
        call_kwargs = mock_col.query.call_args
        assert call_kwargs.kwargs.get("n_results") == 3

    def test_query_vectorstore_empty_collection(self):
        import numpy as np
        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros((1, 384), dtype="float32")
        mock_col = MagicMock()
        mock_col.query.return_value = {"documents": [[]]}
        with patch.object(vs, "SentenceTransformer", return_value=mock_st):
            result = vs.query_vectorstore(mock_col, "query", k=5)
        assert result == []


# ===========================================================================
# TestAgent
# ===========================================================================

def _mock_claude_response(payload: dict) -> MagicMock:
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(payload))]
    return mock_response


class TestAgent:
    def _patch_anthropic(self, payload: dict):
        mock_response = _mock_claude_response(payload)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client)
        return mock_cls, mock_client

    def test_analyse_control_compliant(self):
        payload = {"compliance_level": "compliant", "gaps": [], "recommendations": []}
        mock_cls, _ = self._patch_anthropic(payload)
        with patch.object(agent.anthropic, "Anthropic", mock_cls):
            result = agent.analyse_control("We have full access controls.", ["ISO chunk"])
        assert result["compliance_level"] == "compliant"

    def test_analyse_control_partial(self):
        payload = {
            "compliance_level": "partial",
            "gaps": ["No MFA for privileged users"],
            "recommendations": ["Implement MFA"],
        }
        mock_cls, _ = self._patch_anthropic(payload)
        with patch.object(agent.anthropic, "Anthropic", mock_cls):
            result = agent.analyse_control("We have basic access control.", ["ISO chunk"])
        assert result["compliance_level"] == "partial"
        assert len(result["gaps"]) == 1

    def test_analyse_control_non_compliant(self):
        payload = {
            "compliance_level": "non-compliant",
            "gaps": ["No access control policy", "No audit logging"],
            "recommendations": ["Create policy", "Enable logging"],
        }
        mock_cls, _ = self._patch_anthropic(payload)
        with patch.object(agent.anthropic, "Anthropic", mock_cls):
            result = agent.analyse_control("We have nothing.", ["ISO chunk"])
        assert result["compliance_level"] == "non-compliant"
        assert len(result["gaps"]) == 2

    def test_analyse_control_returns_required_keys(self):
        payload = {
            "compliance_level": "partial",
            "gaps": ["gap1"],
            "recommendations": ["rec1"],
        }
        mock_cls, _ = self._patch_anthropic(payload)
        with patch.object(agent.anthropic, "Anthropic", mock_cls):
            result = agent.analyse_control("control text", ["chunk"])
        assert "compliance_level" in result
        assert "gaps" in result
        assert "recommendations" in result

    def test_analyse_control_uses_haiku_model(self):
        payload = {"compliance_level": "compliant", "gaps": [], "recommendations": []}
        mock_cls, mock_client = self._patch_anthropic(payload)
        with patch.object(agent.anthropic, "Anthropic", mock_cls):
            agent.analyse_control("control", ["chunk"])
        call_kwargs = mock_client.messages.create.call_args
        used_model = call_kwargs.kwargs.get("model") or call_kwargs.args[0]
        assert "haiku" in used_model.lower()

    def test_analyse_control_bad_json_raises(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not valid json {{")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client)
        with patch.object(agent.anthropic, "Anthropic", mock_cls):
            with pytest.raises(ValueError):
                agent.analyse_control("control", ["chunk"])


# ===========================================================================
# TestPipeline
# ===========================================================================

class TestPipeline:
    def _make_controls_dir(self, n: int = 2) -> pathlib.Path:
        tmp = pathlib.Path(tempfile.mkdtemp())
        for i in range(n):
            (tmp / f"control_{i}.txt").write_text(f"Control text {i}")
        return tmp

    def test_run_analysis_returns_dataframe(self):
        import pandas as pd
        controls_dir = self._make_controls_dir(2)
        mock_col = MagicMock()
        mock_result = {"compliance_level": "partial", "gaps": ["g1"], "recommendations": ["r1"]}
        with patch.object(vs, "query_vectorstore", return_value=["chunk"]), \
             patch.object(agent, "analyse_control", return_value=mock_result):
            df = pipeline.run_analysis(controls_dir, mock_col, k=3)
        assert isinstance(df, pd.DataFrame)

    def test_run_analysis_columns(self):
        controls_dir = self._make_controls_dir(1)
        mock_col = MagicMock()
        mock_result = {"compliance_level": "compliant", "gaps": [], "recommendations": []}
        with patch.object(vs, "query_vectorstore", return_value=["c"]), \
             patch.object(agent, "analyse_control", return_value=mock_result):
            df = pipeline.run_analysis(controls_dir, mock_col)
        for col in ("control_file", "compliance_level", "gaps", "recommendations"):
            assert col in df.columns

    def test_run_analysis_row_count_matches_controls(self):
        n = 3
        controls_dir = self._make_controls_dir(n)
        mock_col = MagicMock()
        mock_result = {"compliance_level": "partial", "gaps": [], "recommendations": []}
        with patch.object(vs, "query_vectorstore", return_value=["c"]), \
             patch.object(agent, "analyse_control", return_value=mock_result):
            df = pipeline.run_analysis(controls_dir, mock_col)
        assert len(df) == n

    def test_run_analysis_empty_controls_dir(self):
        import pandas as pd
        controls_dir = pathlib.Path(tempfile.mkdtemp())  # no .txt files
        mock_col = MagicMock()
        with patch.object(vs, "query_vectorstore", return_value=[]), \
             patch.object(agent, "analyse_control", return_value={}):
            df = pipeline.run_analysis(controls_dir, mock_col)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
