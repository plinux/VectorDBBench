import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    HNSWFlavor1,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class MySQLVectorTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option(
            "--username",
            type=str,
            help="MySQL username",
            required=True,
        ),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="MySQL password",
            default=lambda: os.environ.get("MYSQL_PWD", ""),
        ),
    ]
    host: Annotated[
        str,
        click.option(
            "--host",
            type=str,
            help="MySQL host",
            default="127.0.0.1",
            show_default=True,
        ),
    ]
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            help="MySQL port",
            default=3306,
            show_default=True,
        ),
    ]
    database: Annotated[
        str,
        click.option(
            "--database",
            type=str,
            help="MySQL database name",
            required=True,
        ),
    ]
    socket: Annotated[
        str,
        click.option(
            "--socket",
            type=str,
            help="Optional MySQL unix socket path",
            default="",
            show_default=False,
        ),
    ]


class MySQLVectorHNSWTypedDict(MySQLVectorTypedDict, HNSWFlavor1):
    provider: Annotated[
        str,
        click.option(
            "--provider",
            type=click.Choice(["HNSWLIB", "FAISS"], case_sensitive=False),
            help="MySQL vector provider",
            default="HNSWLIB",
            show_default=True,
        ),
    ]
    mode: Annotated[
        str,
        click.option(
            "--mode",
            type=click.Choice(["MEMORY", "EXTERNAL"], case_sensitive=False),
            help="MySQL vector index mode",
            default="MEMORY",
            show_default=True,
        ),
    ]


class MySQLVectorDiskANNTypedDict(MySQLVectorTypedDict):
    max_degree: Annotated[
        int | None,
        click.option(
            "--max-degree",
            type=int,
            help="DiskANN max degree",
        ),
    ]
    build_complexity: Annotated[
        int | None,
        click.option(
            "--build-complexity",
            type=int,
            help="DiskANN build complexity",
        ),
    ]
    search_complexity: Annotated[
        int | None,
        click.option(
            "--search-complexity",
            type=int,
            help="DiskANN search complexity",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MySQLVectorHNSWTypedDict)
def MySQLHNSW(**parameters: Unpack[MySQLVectorHNSWTypedDict]):
    from .config import MySQLVectorConfig, MySQLVectorHNSWConfig

    run(
        db=DB.MySQL,
        db_config=MySQLVectorConfig(
            db_label=parameters["db_label"],
            user_name=parameters["username"],
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            database=parameters["database"],
            socket=parameters["socket"],
        ),
        db_case_config=MySQLVectorHNSWConfig(
            m=parameters["m"],
            efConstruction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            provider=parameters["provider"],
            mode=parameters["mode"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(MySQLVectorDiskANNTypedDict)
def MySQLDiskANN(**parameters: Unpack[MySQLVectorDiskANNTypedDict]):
    from .config import MySQLVectorConfig, MySQLVectorDiskANNConfig

    run(
        db=DB.MySQL,
        db_config=MySQLVectorConfig(
            db_label=parameters["db_label"],
            user_name=parameters["username"],
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            database=parameters["database"],
            socket=parameters["socket"],
        ),
        db_case_config=MySQLVectorDiskANNConfig(
            max_degree=parameters["max_degree"],
            build_complexity=parameters["build_complexity"],
            search_complexity=parameters["search_complexity"],
        ),
        **parameters,
    )
