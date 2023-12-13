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
