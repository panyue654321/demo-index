"""PostgreSQL persistence helpers for DemoIndex trees and global chunks."""

from __future__ import annotations

import os
import hashlib
import uuid
from dataclasses import asdict, dataclass
from typing import Any


SECTION_TABLE_NAME = "document_sections"
CHUNK_TABLE_NAME = "section_chunks"
EMBEDDING_DIMENSION = 1024


@dataclass(frozen=True)
class SectionRecord:
    """One flattened tree node ready for PostgreSQL persistence."""

    section_id: str
    parent_id: str | None
    doc_id: str
    node_id: str
    title: str
    depth: int
    summary: str


@dataclass(frozen=True)
class FlattenedSection:
    """One tree node enriched with traversal metadata for downstream indexing."""

    section_id: str
    parent_id: str | None
    doc_id: str
    node_id: str
    title: str
    depth: int
    summary: str
    title_path: str
    page_index: int | None
    text: str
    is_leaf: bool


@dataclass(frozen=True)
class ChunkRecord:
    """One chunk row ready for PostgreSQL persistence."""

    chunk_id: str
    doc_id: str
    section_id: str
    node_id: str
    chunk_index: int
    title: str
    title_path: str
    page_index: int | None
    chunk_text: str
    search_text: str
    token_count: int
    text_hash: str
    embedding: list[float]


def persist_document_sections(
    tree_payload: dict[str, Any],
    database_url: str | None = None,
) -> dict[str, Any]:
    """Persist one DemoIndex tree payload into PostgreSQL."""
    resolved_database_url = resolve_database_url(database_url)
    records = flatten_document_sections(tree_payload)
    psycopg = _import_psycopg()

    with psycopg.connect(resolved_database_url) as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                _ensure_section_schema(cursor)
                cursor.execute(
                    f"DELETE FROM {SECTION_TABLE_NAME} WHERE doc_id = %s",
                    (records[0].doc_id if records else str(tree_payload.get("doc_id") or ""),),
                )
                if records:
                    cursor.executemany(
                        f"""
                        INSERT INTO {SECTION_TABLE_NAME} (
                            section_id,
                            parent_id,
                            doc_id,
                            node_id,
                            title,
                            depth,
                            summary
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                record.section_id,
                                record.parent_id,
                                record.doc_id,
                                record.node_id,
                                record.title,
                                record.depth,
                                record.summary,
                            )
                            for record in records
                        ],
                    )

    return {
        "table_name": SECTION_TABLE_NAME,
        "doc_id": str(tree_payload.get("doc_id") or ""),
        "row_count": len(records),
        "records": [asdict(record) for record in records[:5]],
    }


def flatten_document_sections(tree_payload: dict[str, Any]) -> list[SectionRecord]:
    """Flatten a DemoIndex output tree into preorder section rows."""
    return [
        SectionRecord(
            section_id=section.section_id,
            parent_id=section.parent_id,
            doc_id=section.doc_id,
            node_id=section.node_id,
            title=section.title,
            depth=section.depth,
            summary=section.summary,
        )
        for section in flatten_tree_sections(tree_payload)
    ]


def flatten_tree_sections(tree_payload: dict[str, Any]) -> list[FlattenedSection]:
    """Flatten a DemoIndex output tree into preorder sections with paths and text."""
    doc_id = str(tree_payload.get("doc_id") or "").strip()
    if not doc_id:
        raise ValueError("Tree payload is missing a top-level doc_id.")

    flattened: list[FlattenedSection] = []

    def walk(
        nodes: list[dict[str, Any]],
        parent_id: str | None,
        depth: int,
        path_titles: list[str],
    ) -> None:
        for node in nodes:
            node_id = str(node.get("node_id") or "").strip()
            title = str(node.get("title") or "").strip()
            summary = str(node.get("summary") or "").strip()
            text = str(node.get("text") or "")
            if not node_id:
                raise ValueError(f"Encountered a node without node_id under doc_id={doc_id}.")
            if not title:
                raise ValueError(f"Encountered node_id={node_id} without title.")
            if not summary:
                raise ValueError(
                    f"Encountered node_id={node_id} without summary. "
                    "Generate summaries before attempting PostgreSQL persistence."
                )
            section_id = build_section_id(doc_id=doc_id, node_id=node_id)
            children = node.get("nodes") or []
            current_path = [*path_titles, title]
            flattened.append(
                FlattenedSection(
                    section_id=section_id,
                    parent_id=parent_id,
                    doc_id=doc_id,
                    node_id=node_id,
                    title=title,
                    depth=depth,
                    summary=summary,
                    title_path=" > ".join(part for part in current_path if part),
                    page_index=_coerce_page_index(node.get("page_index")),
                    text=text,
                    is_leaf=not children,
                )
            )
            walk(children, section_id, depth + 1, current_path)

    walk(tree_payload.get("result") or [], None, 0, [])
    return flattened


def persist_section_chunks(
    chunk_records: list[ChunkRecord],
    *,
    doc_id: str,
    database_url: str | None = None,
) -> dict[str, Any]:
    """Persist one document's global chunk index into PostgreSQL."""
    resolved_database_url = resolve_database_url(database_url)
    psycopg = _import_psycopg()

    with psycopg.connect(resolved_database_url) as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                _ensure_chunk_schema(cursor)
                cursor.execute(
                    f"DELETE FROM {CHUNK_TABLE_NAME} WHERE doc_id = %s",
                    (doc_id,),
                )
                if chunk_records:
                    cursor.executemany(
                        f"""
                        INSERT INTO {CHUNK_TABLE_NAME} (
                            chunk_id,
                            doc_id,
                            section_id,
                            node_id,
                            chunk_index,
                            title,
                            title_path,
                            page_index,
                            chunk_text,
                            search_text,
                            token_count,
                            text_hash,
                            embedding
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector
                        )
                        """,
                        [
                            (
                                record.chunk_id,
                                record.doc_id,
                                record.section_id,
                                record.node_id,
                                record.chunk_index,
                                record.title,
                                record.title_path,
                                record.page_index,
                                record.chunk_text,
                                record.search_text,
                                record.token_count,
                                record.text_hash,
                                _vector_literal(record.embedding),
                            )
                            for record in chunk_records
                        ],
                    )

    return {
        "table_name": CHUNK_TABLE_NAME,
        "doc_id": doc_id,
        "row_count": len(chunk_records),
        "records": [
            {
                "chunk_id": record.chunk_id,
                "section_id": record.section_id,
                "node_id": record.node_id,
                "chunk_index": record.chunk_index,
                "title": record.title,
                "title_path": record.title_path,
                "page_index": record.page_index,
                "token_count": record.token_count,
                "text_hash": record.text_hash,
            }
            for record in chunk_records[:5]
        ],
    }


def resolve_database_url(database_url: str | None = None) -> str:
    """Resolve the PostgreSQL connection string from arguments or environment."""
    resolved_database_url = str(database_url or os.getenv("DATABASE_URL") or "").strip()
    if not resolved_database_url:
        raise RuntimeError("DATABASE_URL is required when PostgreSQL persistence is enabled.")
    return resolved_database_url


def build_section_id(*, doc_id: str, node_id: str) -> str:
    """Build a stable section UUID from doc_id and node_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{node_id}"))


def build_chunk_id(*, doc_id: str, section_id: str, chunk_index: int) -> str:
    """Build a stable chunk UUID from doc_id, section_id, and chunk_index."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{section_id}:{chunk_index}"))


def build_text_hash(text: str) -> str:
    """Build a stable SHA256 hash for chunk text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _import_psycopg():
    """Import psycopg lazily so JSON-only flows do not require it."""
    try:
        import psycopg  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PostgreSQL persistence requires the `psycopg` package. "
            "Install it in the active environment before using PostgreSQL persistence."
        ) from exc
    return psycopg


def _ensure_section_schema(cursor) -> None:
    """Create the document_sections table and supporting indexes when missing."""
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SECTION_TABLE_NAME} (
            section_id UUID PRIMARY KEY,
            parent_id UUID NULL,
            doc_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            title TEXT NOT NULL,
            depth INTEGER NOT NULL,
            summary TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{SECTION_TABLE_NAME}_doc_id "
        f"ON {SECTION_TABLE_NAME} (doc_id)"
    )
    cursor.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{SECTION_TABLE_NAME}_parent_id "
        f"ON {SECTION_TABLE_NAME} (parent_id)"
    )
    cursor.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS uq_{SECTION_TABLE_NAME}_doc_id_node_id "
        f"ON {SECTION_TABLE_NAME} (doc_id, node_id)"
    )


def _ensure_chunk_schema(cursor) -> None:
    """Create the section_chunks table, extensions, and supporting indexes when missing."""
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {CHUNK_TABLE_NAME} (
            chunk_id UUID PRIMARY KEY,
            doc_id TEXT NOT NULL,
            section_id UUID NOT NULL,
            node_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            title TEXT NOT NULL,
            title_path TEXT NOT NULL,
            page_index INTEGER NULL,
            chunk_text TEXT NOT NULL,
            search_text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            text_hash TEXT NOT NULL,
            embedding vector({EMBEDDING_DIMENSION}) NOT NULL
        )
        """
    )
    cursor.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{CHUNK_TABLE_NAME}_doc_id "
        f"ON {CHUNK_TABLE_NAME} (doc_id)"
    )
    cursor.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{CHUNK_TABLE_NAME}_section_id "
        f"ON {CHUNK_TABLE_NAME} (section_id)"
    )
    cursor.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS uq_{CHUNK_TABLE_NAME}_doc_section_chunk "
        f"ON {CHUNK_TABLE_NAME} (doc_id, section_id, chunk_index)"
    )
    cursor.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{CHUNK_TABLE_NAME}_embedding_hnsw "
        f"ON {CHUNK_TABLE_NAME} USING hnsw (embedding vector_cosine_ops)"
    )
    cursor.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{CHUNK_TABLE_NAME}_search_text_trgm "
        f"ON {CHUNK_TABLE_NAME} USING gin (lower(search_text) gin_trgm_ops)"
    )
    cursor.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{CHUNK_TABLE_NAME}_title_trgm "
        f"ON {CHUNK_TABLE_NAME} USING gin (lower(title) gin_trgm_ops)"
    )
    cursor.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{CHUNK_TABLE_NAME}_title_path_trgm "
        f"ON {CHUNK_TABLE_NAME} USING gin (lower(title_path) gin_trgm_ops)"
    )


def _coerce_page_index(value: Any) -> int | None:
    """Convert page indexes to ints when possible."""
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _vector_literal(values: list[float]) -> str:
    """Convert a Python float vector into a PostgreSQL vector literal."""
    if len(values) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(values)}."
        )
    return "[" + ",".join(f"{float(value):.10f}" for value in values) + "]"
