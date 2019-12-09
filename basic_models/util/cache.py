import atexit
import enum
import logging
import os
import pickle
import sqlite3
import threading
import time
from abc import abstractmethod, ABC
from typing import Optional, Any, List, Callable, TypeVar

import MySQLdb

log = logging.getLogger(__name__)

T = TypeVar("T")


class PersistentKeyValueCache(ABC):
    @abstractmethod
    def set(self, key, value):
        """
        Sets a cached value

        :param key: the key under which to store the value
        :param value: the value to store; since None is used indicate the absence of a value, None should not be
            used a value
        """
        pass

    @abstractmethod
    def get(self, key):
        """
        Retrieves a cached value

        :param key: the lookup key
        :return: the cached value or None if no value is found
        """
        pass


def loadPickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dumpPickle(obj, picklePath):
    os.makedirs(os.path.dirname(picklePath), exist_ok=True)
    with open(picklePath, "wb") as f:
        pickle.dump(obj, f)


class PicklePersistentKeyValueCache(PersistentKeyValueCache):
    """
    Represents a key-value cache as a dictionary which is persisted in a file using pickle
    """
    def __init__(self, picklePath, version=1, saveOnUpdate=True, deferredSaveDelaySecs=1.0):
        """
        :param picklePath: the path of the file where the cache values are to be persisted
        :param version: the version of cache entries. If a persisted cache with a non-matching version is found, it
            it is discarded
        :param saveOnUpdate: whether to persist the cache after an update; the cache is saved in a deferred
            manner and will be saved after deferredSaveDelaySecs if no new updates have arrived in the meantime,
            i.e. it will ultimately be saved deferredSaveDelaySecs after the latest update
        :param deferredSaveDelaySecs: the number of seconds to wait for additional data to be added to the cache
            before actually storing the cache after a cache update
        """
        self.deferredSaveDelaySecs = deferredSaveDelaySecs
        self.picklePath = picklePath
        self.version = version
        self.saveOnUpdate = saveOnUpdate
        cacheFound = False
        if os.path.exists(picklePath):
            try:
                persistedVersion, self.cache = loadPickle(picklePath)
                if persistedVersion == version:
                    cacheFound = True
            except EOFError:
                log.warning(f"The cache file in {picklePath} is corrupt")
        if not cacheFound:
            self.cache = {}
        self._saveThread = None
        self._saveThreadSemaphore = threading.Semaphore()

    def save(self):
        """
        Saves the cache in the file whose path was provided at construction
        """
        log.info(f"Saving cache to {self.picklePath}")
        dumpPickle((self.version, self.cache), self.picklePath)

    def get(self, key) -> Optional[Any]:
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value
        if self.saveOnUpdate:
            self._saveDeferred()

    def _saveDeferred(self):
        self._lastUpdateTime = time.time()

        def doSave():
            while True:
                time.sleep(self.deferredSaveDelaySecs)
                timePassedSinceLastUpdate = time.time() - self._lastUpdateTime
                if timePassedSinceLastUpdate >= self.deferredSaveDelaySecs:
                    self.save()
                    return

        if self._saveThread is None or not self._saveThread.is_alive():
            self._saveThreadSemaphore.acquire()
            if self._saveThread is None or not self._saveThread.is_alive():
                self._saveThread = threading.Thread(target=doSave, daemon=False)
                self._saveThread.start()
            self._saveThreadSemaphore.release()


class SqliteConnectionManager:
    _connections: List[sqlite3.Connection] = []
    _atexitHandlerRegistered = False

    @classmethod
    def _registerAtExitHandler(cls):
        if not cls._atexitHandlerRegistered:
            cls._atexitHandlerRegistered = True
            atexit.register(cls._cleanup)

    @classmethod
    def openConnection(cls, path):
        cls._registerAtExitHandler()
        conn = sqlite3.connect(path, check_same_thread=False)
        cls._connections.append(conn)
        return conn

    @classmethod
    def _cleanup(cls):
        for conn in cls._connections:
            conn.close()
        cls._connections = []


class SqlitePersistentKeyValueCache(PersistentKeyValueCache):
    def __init__(self, path, tableName="cache", deferredCommitDelaySecs=1.0):
        self.path = path
        self.conn = SqliteConnectionManager.openConnection(path)
        self.tableName = tableName
        self.maxKeyLength = 255
        self.deferredCommitDelaySecs = deferredCommitDelaySecs
        self._commitThread = None
        self._commitThreadSemaphore = threading.Semaphore()
        self._numEntriesToBeCommitted = 0

        cursor = self.conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table';")
        if tableName not in [r[0] for r in cursor.fetchall()]:
            log.info(f"Creating cache table '{self.tableName}' in {path}")
            cursor.execute(f"CREATE TABLE {tableName} (cache_key VARCHAR({self.maxKeyLength}) PRIMARY KEY, cache_value BLOB);")
        cursor.close()

    def set(self, key, value):
        key = str(key)
        if len(key) > self.maxKeyLength:
            raise ValueError(f"Key too long, maximal key length is {self.maxKeyLength}")
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.tableName} WHERE cache_key=?", (key, ))
        if cursor.fetchone()[0] == 0:
            cursor.execute(f"INSERT INTO {self.tableName} (cache_key, cache_value) VALUES (?, ?)",
                           (key, pickle.dumps(value)))
        else:
            cursor.execute(f"UPDATE {self.tableName} SET cache_value=? WHERE cache_key=?", (pickle.dumps(value), key))
        self._numEntriesToBeCommitted += 1
        self._commitDeferred()
        cursor.close()

    def get(self, key):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT cache_value FROM {self.tableName} WHERE cache_key=?", (str(key), ))
        row = cursor.fetchone()
        if row is None:
            return None
        return pickle.loads(row[0])

    def _commitDeferred(self):
        self._lastUpdateTime = time.time()

        def doCommit():
            while True:
                time.sleep(self.deferredCommitDelaySecs)
                timePassedSinceLastUpdate = time.time() - self._lastUpdateTime
                if timePassedSinceLastUpdate >= self.deferredCommitDelaySecs:
                    log.info(f"Committing {self._numEntriesToBeCommitted} cache entries to the SQLite database {self.path}")
                    self.conn.commit()
                    self._numEntriesToBeCommitted = 0
                    return

        if self._commitThread is None or not self._commitThread.is_alive():
            self._commitThreadSemaphore.acquire()
            if self._commitThread is None or not self._commitThread.is_alive():
                self._commitThread = threading.Thread(target=doCommit, daemon=False)
                self._commitThread.start()
            self._commitThreadSemaphore.release()


class MySQLPersistentKeyValueCache(PersistentKeyValueCache):
    class ValueType(enum.Enum):
        DOUBLE = ("DOUBLE", False)
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
                    log.info(f"Committing {self._numEntriesToBeCommitted} cache entries to the database")
                    self.conn.commit()
                    self._numEntriesToBeCommitted = 0
                    return

        if self._commitThread is None or not self._commitThread.is_alive():
            self._commitThreadSemaphore.acquire()
            if self._commitThread is None or not self._commitThread.is_alive():
                self._commitThread = threading.Thread(target=doCommit, daemon=False)
                self._commitThread.start()
            self._commitThreadSemaphore.release()


class CachedValueProviderMixin(ABC):
    """
    Represents a value provider that can provide values associated with (hashable) keys via a cache or, if
    cached values are not yet present, by computing them
    """
    def __init__(self, cache: PersistentKeyValueCache):
        self._cache = cache

    def _provideValue(self, key, data=None):
        """
        Provides the value for the key by retrieving the associated value from the cache or, if no entry in the
        cache is found, by computing the value via _computeValue

        :param key: the key for which to provide the value
        :param data: optional data required to compute a value
        :return: the retrieved or computed value
        """
        value = self._cache.get(key)
        if value is None:
            value = self._computeValue(key, data)
            self._cache.set(key, value)
        return value

    @abstractmethod
    def _computeValue(self, key, data):
        """
        Computes the value for the given key

        :param key: the key for which to compute the value
        :return: the computed value
        """
        pass


def cached(fn: Callable[[], T], picklePath) -> T:
    if os.path.exists(picklePath):
        log.info(f"Loading cached result from {picklePath}")
        return loadPickle(picklePath)
    else:
        result = fn()
        log.info(f"Saving cached result in {picklePath}")
        dumpPickle(result, picklePath)
        return result

