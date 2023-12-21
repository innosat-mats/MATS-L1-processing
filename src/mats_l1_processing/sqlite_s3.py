import os
from typing import Any, List, Union

from sqlite_s3_query import sqlite_s3_query


class Sqlite_S3_Wrapper:
    def __init__(self, path) -> None:
        self._path = path
        self._query = ""

    def cursor(self) -> "Sqlite_S3_Wrapper":
        return self
    
    def execute(self, query) -> None:
        self._query = query
    
    def fetchall(self) -> Union[Any, List[Any]]:
        with sqlite_s3_query(self._path) as query:
            with query(self._query) as (_, rows):
                return [row for row in rows]


def get_sqlite_connection(filename: str):
    if os.environ.get("MATS_SQLITE_S3", "").upper() == "YES":
        return Sqlite_S3_Wrapper(filename)
    else:
        import sqlite3 as sqlite
        file_conn = sqlite.connect(filename)
        connection = sqlite.connect(':memory:')
        file_conn.backup(connection)
        file_conn.close()
        return connection
