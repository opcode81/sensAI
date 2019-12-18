import atexit
import enum
import logging
import os
import pickle
import threading
import time
from abc import abstractmethod, ABC
from typing import Optional, Any, List, Callable, TypeVar

import sqlite3

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


class DelayedUpdateHook:
    """
    Ensures that a given function is executed after an update happens, but delay the execution until
    there are no further updates for a certain time period
    """
    def __init__(self, fn: Callable[[], Any], timePeriodSecs):
        """
        :param fn: the function to eventually call after an update
        :param timePeriodSecs: the time that must pass while not receiving further updates for fn to be called
        """
        self.fn = fn
        self.timePeriodSecs = timePeriodSecs
        self._lastUpdateTime = None
        self._thread = None
        self._threadLock = threading.Lock()

    def handleUpdate(self):
        """
        Notifies of an update and ensures that the function passed at construction is eventually called
        (after no more updates are received within the respective time window)
        """
        self._lastUpdateTime = time.time()

        def doPeriodicCheck():
            while True:
                time.sleep(self.timePeriodSecs)
                timePassedSinceLastUpdate = time.time() - self._lastUpdateTime
                if timePassedSinceLastUpdate >= self.timePeriodSecs:
                    self.fn()
                    return

        if self._thread is None or not self._thread.is_alive():
            self._threadLock.acquire()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=doPeriodicCheck, daemon=False)
                self._thread.start()
            self._threadLock.release()


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
        self._updateHook = DelayedUpdateHook(self.save, deferredSaveDelaySecs)

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
            self._updateHook.handleUpdate()


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
    class KeyType(enum.Enum):
        STRING = ("VARCHAR(%d)", )
        INTEGER = ("LONG", )

    def __init__(self, path, tableName="cache", deferredCommitDelaySecs=1.0, keyType: KeyType = KeyType.STRING,
            maxKeyLength=255):
        """
        :param path: the path to the file that is to hold the SQLite database
        :param tableName: the name of the table to create in the database
        :param deferredCommitDelaySecs: the time frame during which no new data must be added for a pending transaction to be committed
        :param keyType: the type to use for keys; for complex keys (i.e. tuples), use STRING (conversions to string are automatic)
        :param maxKeyLength: the maximum key length for the case where the keyType can be parametrised (e.g. STRING)
        """
        self.path = path
        self.conn = SqliteConnectionManager.openConnection(path)
        self.tableName = tableName
        self.maxKeyLength = 255
        self.keyType = keyType
        self._updateHook = DelayedUpdateHook(self._commit, deferredCommitDelaySecs)
        self._numEntriesToBeCommitted = 0
        self._connMutex = threading.Lock()

        cursor = self.conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table';")
        if tableName not in [r[0] for r in cursor.fetchall()]:
            log.info(f"Creating cache table '{self.tableName}' in {path}")
            keyDbType = keyType.value[0]
            if "%d" in keyDbType:
                keyDbType = keyDbType % maxKeyLength
            cursor.execute(f"CREATE TABLE {tableName} (cache_key {keyDbType} PRIMARY KEY, cache_value BLOB);")
        cursor.close()

    def _keyDbValue(self, key):
        if self.keyType == self.KeyType.STRING:
            s = str(key)
            if len(s) > self.maxKeyLength:
                raise ValueError(f"Key too long, maximal key length is {self.maxKeyLength}")
            return s
        elif self.keyType == self.KeyType.INTEGER:
            return int(key)
        else:
            raise Exception(f"Unhandled key type {self.keyType}")

    def _commit(self):
        self._connMutex.acquire()
        try:
            log.info(f"Committing {self._numEntriesToBeCommitted} cache entries to the SQLite database {self.path}")
            self.conn.commit()
            self._numEntriesToBeCommitted = 0
        finally:
            self._connMutex.release()

    def set(self, key, value):
        self._connMutex.acquire()
        try:
            cursor = self.conn.cursor()
            key = self._keyDbValue(key)
            cursor.execute(f"SELECT COUNT(*) FROM {self.tableName} WHERE cache_key=?", (key, ))
            if cursor.fetchone()[0] == 0:
                cursor.execute(f"INSERT INTO {self.tableName} (cache_key, cache_value) VALUES (?, ?)",
                               (key, pickle.dumps(value)))
            else:
                cursor.execute(f"UPDATE {self.tableName} SET cache_value=? WHERE cache_key=?", (pickle.dumps(value), key))
            self._numEntriesToBeCommitted += 1
            cursor.close()
        finally:
            self._connMutex.release()

        self._updateHook.handleUpdate()

    def get(self, key):
        self._connMutex.acquire()
        try:
            cursor = self.conn.cursor()
            key = self._keyDbValue(key)
            cursor.execute(f"SELECT cache_value FROM {self.tableName} WHERE cache_key=?", (key, ))
            row = cursor.fetchone()
            cursor.close()
            if row is None:
                return None
            return pickle.loads(row[0])
        finally:
            self._connMutex.release()

    def __len__(self):
        self._connMutex.acquire()
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {self.tableName}")
            cnt = cursor.fetchone()[0]
            cursor.close()
            return cnt
        finally:
            self._connMutex.release()

    def iterItems(self):
        self._connMutex.acquire()
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT cache_key, cache_value FROM {self.tableName}")
            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                yield row[0], pickle.loads(row[1])
            cursor.close()
        finally:
            self._connMutex.release()


class SqlitePersistentList:
    def __init__(self, path):
        self.keyValueCache = SqlitePersistentKeyValueCache(path, keyType=SqlitePersistentKeyValueCache.KeyType.INTEGER)
        self.nextKey = len(self.keyValueCache)

    def append(self, item):
        self.keyValueCache.set(self.nextKey, item)
        self.nextKey += 1

    def iterItems(self):
        for item in self.keyValueCache.iterItems():
            yield item[1]


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
        log.info(f"Loading cached result of function '{fn.__name__}'' from {picklePath}")
        return loadPickle(picklePath)
    else:
        log.info(f"No cached result found in {picklePath}, calling function '{fn.__name__}' ...")
        result = fn()
        log.info(f"Saving cached result in {picklePath}")
        dumpPickle(result, picklePath)
        return result

