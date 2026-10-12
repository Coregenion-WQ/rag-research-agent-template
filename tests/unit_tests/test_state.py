from langchain_core.documents import Document

from shared.state import _generate_uuid, reduce_docs


def test_delete_clears_documents() -> None:
    existing = [Document(page_content="a", metadata={"uuid": "1"})]
    assert reduce_docs(existing, "delete") == []


def test_single_string_is_appended_with_stable_uuid() -> None:
    result = reduce_docs(None, "hello")
    assert len(result) == 1
    assert result[0].page_content == "hello"
    assert result[0].metadata["uuid"] == _generate_uuid("hello")
    assert reduce_docs(None, "hello")[0].metadata["uuid"] == _generate_uuid("hello")


def test_dicts_are_deduplicated_against_existing() -> None:
    existing = reduce_docs(None, [{"page_content": "first"}])
    result = reduce_docs(
        existing,
        [{"page_content": "first"}, {"page_content": "second", "metadata": {"k": 1}}],
    )
    assert [d.page_content for d in result] == ["first", "second"]
    assert result[1].metadata["k"] == 1
    assert result[1].metadata["uuid"] == _generate_uuid("second")


def test_dict_keeps_explicit_uuid_and_id() -> None:
    result = reduce_docs(
        None, [{"page_content": "x", "id": "doc-1", "metadata": {"uuid": "custom"}}]
    )
    assert result[0].id == "doc-1"
    assert result[0].metadata["uuid"] == "custom"


def test_documents_without_uuid_are_not_mutated() -> None:
    original = Document(page_content="text", metadata={"source": "s"})
    result = reduce_docs(None, [original, original])
    assert len(result) == 1
    assert result[0].metadata == {"source": "s", "uuid": _generate_uuid("text")}
    assert "uuid" not in original.metadata


def test_documents_with_uuid_are_kept_as_is() -> None:
    doc = Document(page_content="text", metadata={"uuid": "u1"})
    result = reduce_docs([doc], [doc])
    assert result == [doc]


def test_strings_in_a_list_are_deduplicated() -> None:
    existing = reduce_docs(None, "a")
    result = reduce_docs(existing, ["a", "b", "b"])
    assert [d.page_content for d in result] == ["a", "b"]
    assert result[1].metadata["uuid"] == _generate_uuid("b")
