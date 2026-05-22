import json

from vectordb_bench.backend.clients.api import FilterOp
from vectordb_bench.backend.clients.mysql_vector.mysql_vector import MySQLVector
from vectordb_bench.backend.filter import IntFilter, LabelFilter


class FakeCaseConfig:
    def index_param(self) -> dict:
        return {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "provider": "HNSWLIB",
            "mode": "MEMORY",
            "m": 24,
            "ef_construction": 320,
        }

    def search_param(self) -> dict:
        return {
            "metric_type": "COSINE",
            "ef_search": 96,
        }


class FakeDiskANNCaseConfig:
    def index_param(self) -> dict:
        return {
            "metric_type": "EUCLIDEAN",
            "index_type": "DISKANN",
            "provider": "DISKANN",
            "mode": "EXTERNAL",
            "diskann_max_degree": 40,
            "diskann_build_complexity": 96,
        }

    def search_param(self) -> dict:
        return {
            "metric_type": "EUCLIDEAN",
            "diskann_search_complexity": 128,
        }


class FakeConnection:
    def __init__(self) -> None:
        self.commit_count = 0
        self.rollback_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1


class FakeCursor:
    def __init__(
        self,
        fail_on_stage_call: int | None = None,
        existing_index: bool = False,
        existing_search_ef: int = 96,
    ) -> None:
        self.calls: list[tuple[str, tuple | None]] = []
        self.last_sql = ""
        self.fail_on_stage_call = fail_on_stage_call
        self.stage_call_count = 0
        self.existing_index = existing_index
        self.existing_search_ef = existing_search_ef

    def execute(self, sql: str, params=None) -> None:
        normalized = " ".join(sql.split())
        if normalized == "SELECT VEC_INDEX_STAGE_UPSERT(%s, %s, %s, %s)":
            self.stage_call_count += 1
            if self.fail_on_stage_call == self.stage_call_count:
                raise RuntimeError("stage upsert failed")
        self.calls.append((normalized, params))
        self.last_sql = normalized

    def executemany(self, sql: str, seq_params) -> None:
        normalized = " ".join(sql.split())
        rows = list(seq_params)
        self.calls.append((normalized, tuple(rows)))
        self.last_sql = normalized

    def fetchone(self):
        if self.last_sql == "SELECT VEC_INDEX_TXN_BEGIN()":
            return (42,)
        if self.last_sql == "SELECT VEC_INDEX_INFO(%s)":
            if self.existing_index:
                return (
                    json.dumps(
                        {
                            "lifecycle_state": "ready",
                            "search_ef": self.existing_search_ef,
                        }
                    ),
                )
            return None
        if self.last_sql == "SELECT VEC_INDEX_SEARCH(%s, %s, %s)":
            return (json.dumps([7, 3]),)
        return (1,)

    def fetchall(self):
        if self.last_sql.startswith("SELECT `id` FROM `items`"):
            return [(7,), (3,)]
        return []


def build_client(
    fail_on_stage_call: int | None = None,
    existing_index: bool = False,
    existing_search_ef: int = 96,
    case_config=None,
    database: str = "vectordbbench",
) -> tuple[MySQLVector, FakeConnection, FakeCursor]:
    conn = FakeConnection()
    cursor = FakeCursor(
        fail_on_stage_call=fail_on_stage_call,
        existing_index=existing_index,
        existing_search_ef=existing_search_ef,
    )
    client = MySQLVector.__new__(MySQLVector)
    client.name = "MySQL"
    client.dim = 4
    client.db_config = {
        "user": "root",
        "password": "",
        "host": "127.0.0.1",
        "port": 3306,
        "database": database,
        "unix_socket": None,
    }
    client.case_config = case_config or FakeCaseConfig()
    client.table_name = "items"
    client._vector_field = "embedding"
    client._primary_field = "id"
    client._conn = conn
    client._cursor = cursor
    client._index_ready = False
    client._search_params_ready = False
    client._active_vector_txn_id = None
    client._pending_vector_changes = 0
    client._filter_type = FilterOp.NonFilter
    client._filter_params = ()
    client._scalar_label_field = "labels"
    client._search_sql = "SELECT VEC_INDEX_SEARCH(%s, %s, %s)"
    client.load_batch_size = 2
    return client, conn, cursor


def test_insert_embeddings_uses_binary_stage_upsert_path() -> None:
    client, conn, cursor = build_client()
    embeddings = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ]

    inserted, error = client.insert_embeddings(embeddings=embeddings, metadata=[10, 11, 12])
    client._flush_vector_txn()

    assert inserted == 3
    assert error is None

    executed_sql = [sql for sql, _ in cursor.calls]
    assert "SELECT VEC_INDEX_CREATE(%s, %s, %s, %s, %s)" in executed_sql
    assert "SELECT VEC_INDEX_SET_HNSW_BUILD_PARAMS(%s, %s, %s)" in executed_sql
    assert "SELECT VEC_INDEX_SET_SEARCH_EF(%s, %s)" in executed_sql
    assert executed_sql.count("SELECT VEC_INDEX_STAGE_UPSERT(%s, %s, %s, %s)") == 3
    assert executed_sql.count("SELECT VEC_INDEX_TXN_COMMIT(%s)") == 2

    upsert_params = [
        params
        for sql, params in cursor.calls
        if sql == "SELECT VEC_INDEX_STAGE_UPSERT(%s, %s, %s, %s)"
    ]
    assert [params[2] for params in upsert_params] == [10, 11, 12]
    assert all(isinstance(params[3], bytes) for params in upsert_params)
    assert all(len(params[3]) == 16 for params in upsert_params)
    assert conn.rollback_count == 0


def test_search_embedding_uses_binary_query_payload() -> None:
    client, _, cursor = build_client()
    client._index_ready = True

    result = client.search_embedding(query=[1.0, 2.0, 3.0, 4.0], k=5)

    assert result == [7, 3]
    search_calls = [
        params
        for sql, params in cursor.calls
        if sql == "SELECT VEC_INDEX_SEARCH(%s, %s, %s)"
    ]
    assert len(search_calls) == 1
    assert search_calls[0][0] == "vectordbbench.items.embedding"
    assert isinstance(search_calls[0][1], bytes)
    assert len(search_calls[0][1]) == 16
    assert search_calls[0][2] == 5


def test_insert_embeddings_rolls_back_vector_txn_on_failure() -> None:
    client, conn, cursor = build_client(fail_on_stage_call=2)

    inserted, error = client.insert_embeddings(
        embeddings=[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        metadata=[10, 11],
    )

    assert inserted == 0
    assert error is not None
    executed_sql = [sql for sql, _ in cursor.calls]
    assert "SELECT VEC_INDEX_TXN_ROLLBACK(%s)" in executed_sql
    assert conn.rollback_count == 0


def test_optimize_reuses_existing_index_metadata() -> None:
    client, _, cursor = build_client(existing_index=True, existing_search_ef=32)

    client.optimize(data_size=10)

    executed_sql = [sql for sql, _ in cursor.calls]
    assert "SELECT VEC_INDEX_INFO(%s)" in executed_sql
    assert "SELECT VEC_INDEX_CREATE(%s, %s, %s, %s, %s)" not in executed_sql
    assert "SELECT VEC_INDEX_SET_SEARCH_EF(%s, %s)" in executed_sql


def test_optimize_skips_search_ef_when_existing_index_matches() -> None:
    client, _, cursor = build_client(existing_index=True, existing_search_ef=96)

    client.optimize(data_size=10)

    executed_sql = [sql for sql, _ in cursor.calls]
    assert "SELECT VEC_INDEX_INFO(%s)" in executed_sql
    assert "SELECT VEC_INDEX_SET_SEARCH_EF(%s, %s)" not in executed_sql


def test_diskann_create_and_search_params_use_provider_specific_functions() -> None:
    client, _, cursor = build_client(
        case_config=FakeDiskANNCaseConfig(),
        existing_search_ef=0,
        database="vectordbbench_diskann",
    )

    client.optimize(data_size=10)

    executed_sql = [sql for sql, _ in cursor.calls]
    assert "SELECT VEC_INDEX_CREATE(%s, %s, %s, %s, %s)" in executed_sql
    assert "SELECT VEC_INDEX_SET_DISKANN_BUILD_PARAMS(%s, %s, %s)" in executed_sql
    assert (
        "SELECT VEC_INDEX_SET_DISKANN_SEARCH_COMPLEXITY(%s, %s)" in executed_sql
    )
    assert "SELECT VEC_INDEX_SET_HNSW_BUILD_PARAMS(%s, %s, %s)" not in executed_sql
    assert "SELECT VEC_INDEX_SET_SEARCH_EF(%s, %s)" not in executed_sql


def test_numge_filter_uses_exact_sql_path() -> None:
    client, conn, cursor = build_client()
    client.prepare_filter(IntFilter(filter_rate=0.01, int_value=10))

    inserted, error = client.insert_embeddings(
        embeddings=[[1.0, 2.0, 3.0, 4.0]],
        metadata=[10],
    )
    result = client.search_embedding(query=[1.0, 2.0, 3.0, 4.0], k=2)

    assert inserted == 1
    assert error is None
    assert result == [7, 3]
    executed_sql = [sql for sql, _ in cursor.calls]
    assert "INSERT INTO `items` (`id`, `embedding`) VALUES (%s, VEC_FROMTEXT(%s))" in executed_sql
    assert "SELECT VEC_INDEX_CREATE(%s, %s, %s, %s, %s)" not in executed_sql
    assert any(
        sql.startswith("SELECT `id` FROM `items` WHERE `id` >= %s ORDER BY VEC_DISTANCE")
        for sql in executed_sql
    )
    assert conn.commit_count == 1


def test_label_filter_uses_exact_sql_path() -> None:
    client, _, cursor = build_client()
    client.prepare_filter(LabelFilter(label_percentage=0.01))

    inserted, error = client.insert_embeddings(
        embeddings=[[1.0, 2.0, 3.0, 4.0]],
        metadata=[10],
        labels_data=["label_1p"],
    )
    result = client.search_embedding(query=[1.0, 2.0, 3.0, 4.0], k=2)

    assert inserted == 1
    assert error is None
    assert result == [7, 3]
    executed_sql = [sql for sql, _ in cursor.calls]
    assert (
        "INSERT INTO `items` (`id`, `embedding`, `labels`) VALUES (%s, VEC_FROMTEXT(%s), %s)"
        in executed_sql
    )
    assert any(
        sql.startswith("SELECT `id` FROM `items` WHERE `labels` = %s ORDER BY VEC_DISTANCE")
        for sql in executed_sql
    )
