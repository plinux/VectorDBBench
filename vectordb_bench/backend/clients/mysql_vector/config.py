import os
from typing import TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class MySQLVectorConfigDict(TypedDict):
    """Connection kwargs passed directly to mysql-connector-python."""

    user: str
    password: str
    host: str
    port: int
    database: str
    unix_socket: str | None


class MySQLVectorConfig(DBConfig):
    user_name: str = "root"
    password: SecretStr = SecretStr("")
    host: str = "127.0.0.1"
    port: int = 3306
    database: str = "vectordbbench"
    socket: str = ""

    @staticmethod
    def common_long_configs() -> list[str]:
        return [*DBConfig.common_long_configs(), "password", "socket"]

    def to_dict(self) -> MySQLVectorConfigDict:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user_name,
            "password": self.password.get_secret_value(),
            "database": self.database,
            "unix_socket": self.socket or None,
        }


class MySQLVectorIndexConfig(BaseModel):
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "EUCLIDEAN"
        if self.metric_type == MetricType.COSINE:
            return "COSINE"
        msg = f"Metric type {self.metric_type} is not supported by MySQL vector benchmark client."
        raise ValueError(msg)


class MySQLVectorHNSWConfig(MySQLVectorIndexConfig, DBCaseConfig):
    m: int | None = None
    efConstruction: int | None = None
    ef_search: int | None = None
    index: IndexType = IndexType.HNSW
    provider: str = "HNSWLIB"
    mode: str = "MEMORY"

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "provider": self.provider.upper(),
            "mode": self.mode.upper(),
            "m": self.m,
            "ef_construction": self.efConstruction,
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "ef_search": self.ef_search,
        }


class MySQLVectorDiskANNConfig(MySQLVectorIndexConfig, DBCaseConfig):
    max_degree: int | None = None
    build_complexity: int | None = None
    search_complexity: int | None = None
    index: IndexType = IndexType.DISKANN
    provider: str = "DISKANN"
    mode: str = "EXTERNAL"

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "provider": self.provider.upper(),
            "mode": self.mode.upper(),
            "diskann_max_degree": self.max_degree,
            "diskann_build_complexity": self.build_complexity,
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "diskann_search_complexity": self.search_complexity,
        }


_mysql_vector_case_config = {
    IndexType.HNSW: MySQLVectorHNSWConfig,
    IndexType.DISKANN: MySQLVectorDiskANNConfig,
}
