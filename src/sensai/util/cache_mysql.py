import enum
import logging
import pickle
import threading
import time

import MySQLdb

from .cache import PersistentKeyValueCache


_log = logging.getLogger(__name__)


class MySQLPersistentKeyValueCache(PersistentKeyValueCache):
    class ValueType(enum.Enum):
        DOUBLE = ("DOUBLE", False)  # (SQL data type, isCachedValuePickled)
        BLOB = ("BLOB", True)

    def __init__(self, host, db, user, pw, valueType: ValueType, tableName="cache", deferredCommitDelaySecs=1.0):
        self.conn = MySQLdb.connect(host, db, user, pw)
        self.tableName = tableName
        self.maxKeyLength = 255
        self.deferredCommitDelaySecs = deferredCommitDelaySecs
        self._commitThread = None
        self._commitThreadSemaphore = threading.Semaphore()
        self._numEntriesToBeCommitted = 0

        cacheValueSqlType, self.isCacheValuePickled = valueType.value

        cursor = self.conn.cursor()
        cursor.execute(f"SHOW TABLES;")
        if tableName not in [r[0] for r in cursor.fetchall()]:
            cursor.execute(f"CREATE TABLE {tableName} (cache_key VARCHAR({self.maxKeyLength}) PRIMARY KEY, cache_value {cacheValueSqlType});")
        cursor.close()

    def set(self, key, value):
        key = str(key)
        if len(key) > self.maxKeyLength:
            raise ValueError(f"Key too long, maximal key length is {self.maxKeyLength}")
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.tableName} WHERE cache_key=%s", (key, ))
        storedValue = pickle.dumps(value) if self.isCacheValuePickled else value
        if cursor.fetchone()[0] == 0:
            cursor.execute(f"INSERT INTO {self.tableName} (cache_key, cache_value) VALUES (%s, %s)",
                (key, storedValue))
        else:
            cursor.execute(f"UPDATE {self.tableName} SET cache_value=%s WHERE cache_key=%s", (storedValue, key))
        self._numEntriesToBeCommitted += 1
        self._commitDeferred()
        #self.conn.commit()
        cursor.close()

    def get(self, key):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT cache_value FROM {self.tableName} WHERE cache_key=%s", (str(key), ))
        row = cursor.fetchone()
        if row is None:
            return None
        storedValue = row[0]
        value = pickle.loads(storedValue) if self.isCacheValuePickled else storedValue
        return value

    def _commitDeferred(self):
        self._lastUpdateTime = time.time()

        def doCommit():
            while True:
                time.sleep(self.deferredCommitDelaySecs)
                timePassedSinceLastUpdate = time.time() - self._lastUpdateTime
                if timePassedSinceLastUpdate >= self.deferredCommitDelaySecs:
                    _log.info(f"Committing {self._numEntriesToBeCommitted} cache entries to the database")
                    self.conn.commit()
                    self._numEntriesToBeCommitted = 0
                    return

        if self._commitThread is None or not self._commitThread.is_alive():
            self._commitThreadSemaphore.acquire()
            if self._commitThread is None or not self._commitThread.is_alive():
                self._commitThread = threading.Thread(target=doCommit, daemon=False)
                self._commitThread.start()
            self._commitThreadSemaphore.release()
