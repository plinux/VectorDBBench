import json
import logging
import struct
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import mysql.connector as mysql

from ..api import FilterOp, VectorDB
from ...filter import Filter
from .config import MySQLVectorConfigDict, MySQLVectorHNSWConfig

log = logging.getLogger(__name__)

MYSQL_VECTOR_DEFAULT_LOAD_BATCH_SIZE = 2048
MYSQL_VECTOR_DEFAULT_HNSW_M = 16
MYSQL_VECTOR_DEFAULT_HNSW_EF_CONSTRUCTION = 200
MYSQL_VECTOR_DEFAULT_DISKANN_MAX_DEGREE = 32
MYSQL_VECTOR_DEFAULT_DISKANN_BUILD_COMPLEXITY = 64
MYSQL_VECTOR_DEFAULT_DISKANN_SEARCH_COMPLEXITY = 64


class MySQLVector(VectorDB):
    """Benchmark wrapper for MySQL named vector indexes."""

    thread_safe: bool = False
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: MySQLVectorConfigDict,
        db_case_config: MySQLVectorHNSWConfig,
        collection_name: str = "items",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "MySQL"
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self._vector_field = "embedding"
        self._primary_field = "id"
        self._scalar_label_field = "labels"
        self._conn = None
        self._cursor = None
        self._index_ready = False
        self._search_params_ready = False
        self._active_vector_txn_id = None
        self._pending_vector_changes = 0
        self._filter_type = FilterOp.NonFilter
        self._filter_params: tuple[Any, ...] = ()
        self._search_sql = "SELECT VEC_INDEX_SEARCH(%s, %s, %s)"
        self.load_batch_size = MYSQL_VECTOR_DEFAULT_LOAD_BATCH_SIZE

        self._connect(with_database=False)
        try:
            self._ensure_database()
            self._reconnect_for_database()
            if drop_old:
                self._drop_index()
                self._drop_table()
            self._ensure_table()
        finally:
            self._disconnect()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_conn"] = None
        state["_cursor"] = None
        state["_active_vector_txn_id"] = None
        state["_pending_vector_changes"] = 0
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        self._conn = None
        self._cursor = None
        self._active_vector_txn_id = None
        self._pending_vector_changes = 0

    @property
    def index_name(self) -> str:
        return f"{self.db_config['database']}.{self.table_name}.{self._vector_field}"

    def _connect(self, with_database: bool = True) -> None:
        connect_kwargs = {
            "user": self.db_config["user"],
            "password": self.db_config["password"],
        }
        unix_socket = self.db_config.get("unix_socket")
        if unix_socket:
            connect_kwargs["unix_socket"] = unix_socket
        else:
            connect_kwargs["host"] = self.db_config["host"]
            connect_kwargs["port"] = self.db_config["port"]
        if with_database:
            connect_kwargs["database"] = self.db_config["database"]
        self._conn = mysql.connect(**connect_kwargs)
        self._conn.autocommit = False
        self._cursor = self._conn.cursor()

    def _reconnect_for_database(self) -> None:
        self._disconnect()
        self._connect(with_database=True)

    def _disconnect(self) -> None:
        if self._cursor is not None and self._conn is not None:
            try:
                self._flush_vector_txn()
            except Exception:  # noqa: BLE001
                self._conn.rollback()
                self._active_vector_txn_id = None
                self._pending_vector_changes = 0
        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _ensure_database(self) -> None:
        assert self._cursor is not None
        self._cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.db_config['database']}`")
        self._conn.commit()

    def _drop_table(self) -> None:
        assert self._cursor is not None
        self._cursor.execute(f"DROP TABLE IF EXISTS `{self.table_name}`")
        self._conn.commit()

    def _drop_index(self) -> None:
        assert self._cursor is not None
        try:
            self._cursor.execute("SELECT VEC_INDEX_DROP(%s)", (self.index_name,))
            self._cursor.fetchone()
            self._conn.commit()
        except Exception:  # noqa: BLE001
            self._conn.rollback()

    def _ensure_table(self) -> None:
        assert self._cursor is not None
        self._cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.table_name}` (
                `{self._primary_field}` BIGINT NOT NULL,
                `{self._vector_field}` VECTOR({self.dim}) NOT NULL,
                `{self._scalar_label_field}` VARCHAR(128) NULL,
                PRIMARY KEY (`{self._primary_field}`)
            ) ENGINE=InnoDB
            """
        )
        self._conn.commit()
        self._index_ready = False
        self._search_params_ready = False

    def _create_index(self) -> None:
        assert self._cursor is not None
        index_param = self.case_config.index_param()
        self._cursor.execute(
            "SELECT VEC_INDEX_CREATE(%s, %s, %s, %s, %s)",
            (
                self.index_name,
                self.dim,
                index_param["metric_type"],
                index_param["mode"],
                index_param["provider"],
            ),
        )
        self._cursor.fetchone()
        self._conn.commit()
        self._apply_build_params(index_param)
        self._index_ready = True
        self._apply_search_params()

    def _apply_build_params(self, index_param: dict) -> None:
        assert self._cursor is not None
        provider = index_param["provider"]

        if provider == "HNSWLIB":
            requested_hnsw_m = (
                index_param["m"]
                if index_param["m"] is not None
                else MYSQL_VECTOR_DEFAULT_HNSW_M
            )
            requested_hnsw_ef_construction = (
                index_param["ef_construction"]
                if index_param["ef_construction"] is not None
                else MYSQL_VECTOR_DEFAULT_HNSW_EF_CONSTRUCTION
            )

            if (
                requested_hnsw_m,
                requested_hnsw_ef_construction,
            ) == (
                MYSQL_VECTOR_DEFAULT_HNSW_M,
                MYSQL_VECTOR_DEFAULT_HNSW_EF_CONSTRUCTION,
            ):
                return

            self._cursor.execute(
                "SELECT VEC_INDEX_SET_HNSW_BUILD_PARAMS(%s, %s, %s)",
                (
                    self.index_name,
                    requested_hnsw_m,
                    requested_hnsw_ef_construction,
                ),
            )
            self._cursor.fetchone()
            self._conn.commit()
            return

        if provider != "DISKANN":
            return

        requested_max_degree = (
            index_param["diskann_max_degree"]
            if index_param["diskann_max_degree"] is not None
            else MYSQL_VECTOR_DEFAULT_DISKANN_MAX_DEGREE
        )
        requested_build_complexity = (
            index_param["diskann_build_complexity"]
            if index_param["diskann_build_complexity"] is not None
            else MYSQL_VECTOR_DEFAULT_DISKANN_BUILD_COMPLEXITY
        )

        if (
            requested_max_degree,
            requested_build_complexity,
        ) == (
            MYSQL_VECTOR_DEFAULT_DISKANN_MAX_DEGREE,
            MYSQL_VECTOR_DEFAULT_DISKANN_BUILD_COMPLEXITY,
        ):
            return

        self._cursor.execute(
            "SELECT VEC_INDEX_SET_DISKANN_BUILD_PARAMS(%s, %s, %s)",
            (
                self.index_name,
                requested_max_degree,
                requested_build_complexity,
            ),
        )
        self._cursor.fetchone()
        self._conn.commit()

    def _apply_search_params(self) -> None:
        assert self._cursor is not None
        if self._search_params_ready:
            return

        search_param = self.case_config.search_param()
        if search_param.get("ef_search") is not None:
            self._cursor.execute(
                "SELECT VEC_INDEX_SET_SEARCH_EF(%s, %s)",
                (self.index_name, search_param["ef_search"]),
            )
            self._cursor.fetchone()
            self._conn.commit()

        if search_param.get("diskann_search_complexity") is not None:
            self._cursor.execute(
                "SELECT VEC_INDEX_SET_DISKANN_SEARCH_COMPLEXITY(%s, %s)",
                (self.index_name, search_param["diskann_search_complexity"]),
            )
            self._cursor.fetchone()
            self._conn.commit()

        self._search_params_ready = True

    def _attach_existing_index(self) -> bool:
        assert self._cursor is not None
        try:
            self._cursor.execute("SELECT VEC_INDEX_INFO(%s)", (self.index_name,))
            row = self._cursor.fetchone()
        except Exception:  # noqa: BLE001
            self._conn.rollback()
            return False

        if row is None or row[0] is None:
            return False

        info = json.loads(row[0])
        search_param = self.case_config.search_param()
        desired_search_ef = search_param.get("ef_search")
        desired_diskann_search_complexity = search_param.get(
            "diskann_search_complexity"
        )

        self._index_ready = True
        self._search_params_ready = (
            (
                desired_search_ef is None
                or info.get("search_ef") == desired_search_ef
            )
            and (
                desired_diskann_search_complexity is None
                or info.get("diskann_search_complexity")
                == desired_diskann_search_complexity
            )
        )
        self._apply_search_params()
        return True

    def _ensure_index_ready(self) -> None:
        if self._index_ready:
            self._apply_search_params()
            return
        if self._attach_existing_index():
            return
        self._create_index()

    def _pack_vector_payload(self, vector: list[float]) -> bytes:
        return struct.pack(f"<{len(vector)}f", *vector)

    def _json_vector(self, vector: list[float]) -> str:
        return json.dumps(vector, separators=(",", ":"))

    def _uses_exact_filter_search(self) -> bool:
        return self._filter_type != FilterOp.NonFilter

    def _begin_vector_txn(self) -> int:
        assert self._cursor is not None
        self._cursor.execute("SELECT VEC_INDEX_TXN_BEGIN()")
        row = self._cursor.fetchone()
        if row is None:
            raise RuntimeError("Failed to start vector transaction")
        return int(row[0])

    def _commit_vector_txn(self, txn_id: int) -> None:
        assert self._cursor is not None
        self._cursor.execute("SELECT VEC_INDEX_TXN_COMMIT(%s)", (txn_id,))
        self._cursor.fetchone()
        self._conn.commit()

    def _rollback_vector_txn(self, txn_id: int) -> None:
        assert self._cursor is not None
        self._cursor.execute("SELECT VEC_INDEX_TXN_ROLLBACK(%s)", (txn_id,))
        self._cursor.fetchone()
        self._conn.commit()

    def _ensure_vector_txn(self) -> int:
        if self._active_vector_txn_id is None:
            self._active_vector_txn_id = self._begin_vector_txn()
            self._pending_vector_changes = 0
        return self._active_vector_txn_id

    def _flush_vector_txn(self) -> None:
        if self._active_vector_txn_id is None:
            return

        txn_id = self._active_vector_txn_id
        if self._pending_vector_changes > 0:
            self._commit_vector_txn(txn_id)
        else:
            self._rollback_vector_txn(txn_id)

        self._active_vector_txn_id = None
        self._pending_vector_changes = 0

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        self._connect(with_database=True)
        try:
            yield
        finally:
            self._disconnect()

    def need_normalize_cosine(self) -> bool:
        return False

    def prepare_filter(self, filters: Filter):
        self._filter_type = filters.type
        self._filter_params = ()

        if filters.type == FilterOp.NonFilter:
            return

        if filters.type == FilterOp.NumGE:
            self._filter_params = (filters.int_value,)
            return

        if filters.type == FilterOp.StrEqual:
            self._filter_params = (filters.label_value,)
            return

        msg = f"Unsupported filter for MySQL: {filters}"
        raise ValueError(msg)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        assert self._cursor is not None
        if self._uses_exact_filter_search():
            return self._insert_table_rows(embeddings, metadata, labels_data)

        self._ensure_index_ready()
        try:
            inserted = 0

            for doc_id, embedding in zip(metadata, embeddings):
                txn_id = self._ensure_vector_txn()
                self._cursor.execute(
                    "SELECT VEC_INDEX_STAGE_UPSERT(%s, %s, %s, %s)",
                    (
                        self.index_name,
                        txn_id,
                        int(doc_id),
                        self._pack_vector_payload(embedding),
                    ),
                )
                self._cursor.fetchone()
                self._pending_vector_changes += 1
                inserted += 1

                if self._pending_vector_changes < self.load_batch_size:
                    continue

                self._flush_vector_txn()

            return inserted, None
        except Exception as exc:  # noqa: BLE001
            if self._active_vector_txn_id is not None:
                try:
                    self._rollback_vector_txn(self._active_vector_txn_id)
                except Exception:  # noqa: BLE001
                    self._conn.rollback()
                finally:
                    self._active_vector_txn_id = None
                    self._pending_vector_changes = 0
            else:
                self._conn.rollback()
            log.warning("Failed to insert MySQL vector embeddings: %s", exc)
            return 0, exc

    def _insert_table_rows(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None,
    ) -> tuple[int, Exception | None]:
        assert self._cursor is not None

        if self._filter_type == FilterOp.StrEqual and labels_data is None:
            msg = "labels_data required for label filter inserts"
            raise ValueError(msg)

        if self._filter_type == FilterOp.StrEqual:
            insert_sql = (
                f"INSERT INTO `{self.table_name}` "
                f"(`{self._primary_field}`, `{self._vector_field}`, `{self._scalar_label_field}`) "
                "VALUES (%s, VEC_FROMTEXT(%s), %s)"
            )
            batch_rows = [
                (int(doc_id), self._json_vector(embedding), labels_data[idx])
                for idx, (doc_id, embedding) in enumerate(zip(metadata, embeddings))
            ]
        else:
            insert_sql = (
                f"INSERT INTO `{self.table_name}` "
                f"(`{self._primary_field}`, `{self._vector_field}`) "
                "VALUES (%s, VEC_FROMTEXT(%s))"
            )
            batch_rows = [
                (int(doc_id), self._json_vector(embedding))
                for doc_id, embedding in zip(metadata, embeddings)
            ]

        try:
            self._cursor.executemany(insert_sql, batch_rows)
            self._conn.commit()
            return len(batch_rows), None
        except Exception as exc:  # noqa: BLE001
            self._conn.rollback()
            log.warning("Failed to insert MySQL filtered benchmark rows: %s", exc)
            return 0, exc

    def optimize(self, data_size: int | None = None):
        assert self._cursor is not None
        if self._uses_exact_filter_search():
            return
        self._flush_vector_txn()
        self._ensure_index_ready()

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs,
    ) -> list[int]:
        assert self._cursor is not None
        if self._uses_exact_filter_search():
            return self._search_embedding_with_exact_filter(query, k)

        self._flush_vector_txn()
        self._ensure_index_ready()
        self._cursor.execute(
            self._search_sql,
            (self.index_name, self._pack_vector_payload(query), k),
        )
        row = self._cursor.fetchone()
        if row is None or row[0] is None:
            return []
        result = json.loads(row[0])
        return [int(doc_id) for doc_id in result]

    def _search_embedding_with_exact_filter(
        self,
        query: list[float],
        k: int,
    ) -> list[int]:
        assert self._cursor is not None
        metric_name = self.case_config.search_param()["metric_type"]
        where_clause = ""

        if self._filter_type == FilterOp.NumGE:
            where_clause = f"WHERE `{self._primary_field}` >= %s"
        elif self._filter_type == FilterOp.StrEqual:
            where_clause = f"WHERE `{self._scalar_label_field}` = %s"

        search_sql = (
            f"SELECT `{self._primary_field}` "
            f"FROM `{self.table_name}` "
            f"{where_clause} "
            f"ORDER BY VEC_DISTANCE(`{self._vector_field}`, VEC_FROMTEXT(%s), %s), `{self._primary_field}` "
            "LIMIT %s"
        )
        self._cursor.execute(
            search_sql,
            (*self._filter_params, self._json_vector(query), metric_name, k),
        )
        return [int(row[0]) for row in self._cursor.fetchall()]
