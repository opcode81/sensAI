import atexit
import enum
import glob
import logging
import os
import pickle
import re
import threading
import time
from abc import abstractmethod, ABC
from typing import Any, Callable, Iterator, List, Optional, TypeVar

import sqlite3

from .pickle import PickleFailureDebugger

_log = logging.getLogger(__name__)

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


class PersistentList(ABC):
    @abstractmethod
    def append(self, item):
        """
        Adds an item to the cache

        :param item: the item to store
        """
        pass

    @abstractmethod
    def iterItems(self):
        """
        Iterates over the items in the persisted list

        :return: generator of item
        """
        pass


def loadPickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dumpPickle(obj, picklePath):
    dirName = os.path.dirname(picklePath)
    if dirName != "":
        os.makedirs(dirName, exist_ok=True)
    with open(picklePath, "wb") as f:
        try:
            pickle.dump(obj, f)
        except AttributeError as e:
            failingPaths = PickleFailureDebugger.debugFailure(obj)
            raise AttributeError(f"Cannot pickle paths {failingPaths} of {obj}: {str(e)}")


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
                _log.info(f"Loading cache from {picklePath}")
                persistedVersion, self.cache = loadPickle(picklePath)
                if persistedVersion == version:
                    cacheFound = True
            except EOFError:
                _log.warning(f"The cache file in {picklePath} is corrupt")
        if not cacheFound:
            self.cache = {}
        self._updateHook = DelayedUpdateHook(self.save, deferredSaveDelaySecs)

    def save(self):
        """
        Saves the cache in the file whose path was provided at construction
        """
        _log.info(f"Saving cache to {self.picklePath}")
        dumpPickle((self.version, self.cache), self.picklePath)

    def get(self, key) -> Optional[Any]:
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value
        if self.saveOnUpdate:
            self._updateHook.handleUpdate()


class SlicedPicklePersistentList(PersistentList):
    """
    Object handling the creation and access to sliced pickle caches
    """
    def __init__(self, directory, pickleBaseName, numEntriesPerSlice=100000):
        """
        :param directory: path to the directory where the sliced caches are to be stored
        :param pickleBaseName: base name for the pickle, where slices will have the names {pickleBaseName}_sliceX.pickle
        :param numEntriesPerSlice: how many entries should be stored in each cache
        """
        self.directory = directory
        self.pickleBaseName = pickleBaseName
        self.numEntriesPerSlice = numEntriesPerSlice

        # Set up the variables for the sliced cache
        self.sliceId = 0
        self.indexInSlice = 0
        self.cacheOfSlice = []

        # Search directory for already present sliced caches
        self.slicedFiles = self._findSlicedCaches()

        # Helper variable to ensure object is only modified within a with-clause
        self._currentlyInWithClause = False

    def __enter__(self):
        self._currentlyInWithClause = True
        if self.cacheExists():
            # Reset state to enable the appending of more items to the cache
            self._setLastCacheState()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dump()
        self._currentlyInWithClause = False

    def append(self, item):
        """
        Append item to cache
        :param item: entry in the cache
        """
        if not self._currentlyInWithClause:
            raise Exception("Class needs to be instantiated within a with-clause to ensure correct storage")

        if (self.indexInSlice + 1) % self.numEntriesPerSlice == 0:
            self._dump()

        self.cacheOfSlice.append(item)
        self.indexInSlice += 1

    def iterItems(self) -> Iterator[Any]:
        """
        Iterate over entries in the sliced cache
        :return: iterator over all items in the cache
        """
        for filePath in self.slicedFiles:
            _log.info(f"Loading sliced pickle list from {filePath}")
            cachedPickle = self._loadPickle(filePath)
            for item in cachedPickle:
                yield item

    def clear(self):
        """
        Clears the cache if it exists
        """
        if self.cacheExists():
            for filePath in self.slicedFiles:
                os.unlink(filePath)

    def cacheExists(self) -> bool:
        """
        Does this cache already exist
        :return: True if cache exists, False if not
        """
        return len(self.slicedFiles) > 0

    def _setLastCacheState(self):
        """
        Sets the state such as to be able to add items to an existant cache
        """
        _log.info("Resetting last state of cache...")
        self.sliceId = len(self.slicedFiles) - 1
        self.cacheOfSlice = self._loadPickle(self._picklePath(self.sliceId))
        self.indexInSlice = len(self.cacheOfSlice) - 1
        if self.indexInSlice >= self.numEntriesPerSlice:
            self._nextSlice()

    def _dump(self):
        """
        Dumps the current cache (if non-empty)
        """
        if len(self.cacheOfSlice) > 0:
            picklePath = self._picklePath(str(self.sliceId))
            _log.info(f"Saving sliced cache to {picklePath}")
            dumpPickle(self.cacheOfSlice, picklePath)
            self.slicedFiles.append(picklePath)

            # Update slice number and reset indexing and cache
            self._nextSlice()
        else:
            _log.warning("Unexpected behavior: Dump was called when cache of slice is 0!")

    def _nextSlice(self):
        """
        Updates sliced cache state for the next slice
        """
        self.sliceId += 1
        self.indexInSlice = 0
        self.cacheOfSlice = []

    def _findSlicedCaches(self) -> List[str]:
        """
        Finds all pickled slices associated with this cache
        :return: list of sliced pickled files
        """
        # glob.glob permits the usage of unix-style pathnames matching. (below we find all ..._slice*.pickle files)
        listOfFileNames = glob.glob(self._picklePath("*"))
        # Sort the slices to ensure it is in the same order as they was produced (regex replaces everything not a number with empty string).
        listOfFileNames.sort(key=lambda f: int(re.sub('\D', '', f)))
        return listOfFileNames

    def _loadPickle(self, picklePath: str) -> List[Any]:
        """
        Loads pickle if file path exists, and persisted version is correct.
        :param picklePath: file path
        :return: list with objects
        """
        cachedPickle = []
        if os.path.exists(picklePath):
            try:
                cachedPickle = loadPickle(picklePath)
            except EOFError:
                _log.warning(f"The cache file in {picklePath} is corrupt")
        else:
            raise Exception(f"The file {picklePath} does not exist!")
        return cachedPickle

    def _picklePath(self, sliceSuffix) -> str:
        return f"{os.path.join(self.directory, self.pickleBaseName)}_slice{sliceSuffix}.pickle"


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
            _log.info(f"Creating cache table '{self.tableName}' in {path}")
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
            _log.info(f"Committing {self._numEntriesToBeCommitted} cache entries to the SQLite database {self.path}")
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

    def _execute(self, cursor, *query):
        try:
            cursor.execute(*query)
        except sqlite3.DatabaseError as e:
            raise Exception(f"Error executing query for {self.path}: {e}")

    def get(self, key):
        self._connMutex.acquire()
        try:
            cursor = self.conn.cursor()
            key = self._keyDbValue(key)
            self._execute(cursor, f"SELECT cache_value FROM {self.tableName} WHERE cache_key=?", (key, ))
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


class SqlitePersistentList(PersistentList):
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
    cached values are not yet present, by computing them.
    """
    def __init__(self, cache: Optional[PersistentKeyValueCache], persistCache=False):
        """


        Args:
            cache: The cache to use or None. Important: when None, caching will be disabled
            persistCache: Whether to persist the cache when pickling
        """
        self.persistCache = persistCache
        self._cache = cache

    def __getstate__(self):
        if not self.persistCache:
            d = self.__dict__.copy()
            d["_cache"] = None
            return d
        return self.__dict__

    def _provideValue(self, key, data=None):
        """
        Provides the value for the key by retrieving the associated value from the cache or, if no entry in the
        cache is found, by computing the value via _computeValue

        :param key: the key for which to provide the value
        :param data: optional data required to compute a value
        :return: the retrieved or computed value
        """
        if self._cache is None:
            return self._computeValue(key, data)
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


def cached(fn: Callable[[], T], picklePath, functionName=None, validityCheckFn: Optional[Callable[[T], bool]] = None) -> T:
    """
    :param fn: the function whose result is to be cached
    :param picklePath: the path in which to store the cached result
    :param functionName: the name of the function fn (for the case where its __name__ attribute is not
        informative)
    :param validityCheckFn: an optional function to call in order to check whether a cached result is still valid;
        the function shall return True if the res is still valid and false otherwise. If a cached result is invalid,
        the function fn is called to compute the result and the cached result is updated.
    :return: the res (either obtained from the cache or the function)
    """
    if functionName is None:
        functionName = fn.__name__

    def callFnAndCacheResult():
        res = fn()
        _log.info(f"Saving cached res in {picklePath}")
        dumpPickle(res, picklePath)
        return res

    if os.path.exists(picklePath):
        _log.info(f"Loading cached res of function '{functionName}' from {picklePath}")
        result = loadPickle(picklePath)
        if validityCheckFn is not None:
            if not validityCheckFn(result):
                _log.info(f"Cached result is no longer valid, recomputing ...")
                result = callFnAndCacheResult()
        return result
    else:
        _log.info(f"No cached res found in {picklePath}, calling function '{functionName}' ...")
        return callFnAndCacheResult()


class PickleCached(object):
    def __init__(self, cacheBasePath: str, filenamePrefix: str = None, filename: str = None):
        """

        :param cacheBasePath:
        :param filenamePrefix:
        :param filename:
        """
        self.filename = filename
        self.cacheBasePath = cacheBasePath
        self.filenamePrefix = filenamePrefix

        if self.filenamePrefix is None:
            self.filenamePrefix = ""
        else:
            self.filenamePrefix += "-"

    def __call__(self, fn, *args, **kwargs):
        if self.filename is None:
            self.filename = self.filenamePrefix + fn.__qualname__ + ".cache.pickle"
        picklePath = os.path.join(self.cacheBasePath,  self.filename)
        return lambda *args, **kwargs: cached(lambda: fn(*args, **kwargs), picklePath, functionName=fn.__name__)
