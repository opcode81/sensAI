from __future__ import annotations
import collections
import functools
import pickle
from abc import ABC, abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Callable, Dict, Union, Any, List, Sequence, Generator, Optional
import json
import logging
import re
import threading


from azure.storage.table import TableService, TableBatch, Entity
from azure.storage.blob import BlockBlobService
import pandas as pd

from .cache import PersistentKeyValueCache, PeriodicUpdateHook

AZURE_ALLOWED_TABLE_NAME_PATTERN = re.compile("^[A-Za-z][A-Za-z0-9]{2,62}$")
AZURE_ALLOWED_TABLE_BATCH_SIZE = 100

_log = logging.getLogger(__name__)


class AzureLazyBatchCommitTable:
    """
    Wrapper for an Azure table, which allow for convenient insertion via lazy batch execution per partition.
    Uses a priority queue to manage order of partitions to be committed.
    To execute insertions, call :func:`LazyBatchCommitTable.commit`
    """

    class PartitionCommandsPriorityQueue:

        class PartitionCommands:
            def __init__(self, partitionKey):
                self.partitionKey = partitionKey
                self._commandList = collections.deque()

            def __len__(self):
                return len(self._commandList)

            def append(self, command):
                self._commandList.append(command)

            def execute(self, contextManager: Callable[[], TableBatch],  batchSize: int):
                while len(self._commandList) > 0:
                    _slice = [self._commandList.popleft() for _ in range(min(batchSize, len(self._commandList)))]
                    _log.info(f"Committing {len(_slice)} cache entries to the database")
                    with contextManager() as batch:
                        for command in _slice:
                            command(batch)

        def __init__(self):
            self.partitionCommandsQueue = []
            self.partitionKey2Commands = {}
            self._threadLock = threading.Lock()

        def addCommand(self, partitionKey, command: Union[Callable[[TableBatch], Any], functools.partial[TableBatch]]):
            """
            Add a command to queue of corresponding partitionKey
            :param partitionKey:
            :param command: a callable on a TableBatch
            """
            with self._threadLock:
                if partitionKey not in self.partitionKey2Commands:
                    commands = self.PartitionCommands(partitionKey)
                    self.partitionCommandsQueue.append(commands)
                    self.partitionKey2Commands[partitionKey] = commands
                self.partitionKey2Commands[partitionKey].append(command)

        def pop(self, minLength: int = None) -> Optional[AzureLazyBatchCommitTable.PartitionCommandsPriorityQueue.PartitionCommands]:
            """
            :param minLength: minimal length of largest PartitionCommands for the pop to take place.
            :return: largest PartitionCommands or None if minimal length is not reached
            """
            with self._threadLock:
                return self._pop(minLength)

        def popAll(self):
            with self._threadLock:
                commandsList = []
                while not self._isEmpty():
                    commandsList.append(self._pop())
                return commandsList

        def isEmpty(self):
            with self._threadLock:
                return self._isEmpty()

        def _pop(self, minLength=None):
            length, index = self._getMaxPriorityInfo()
            if minLength is None or length >= minLength:
                q = self.partitionCommandsQueue.pop(index)
                del self.partitionKey2Commands[q.partitionKey]
                return q
            else:
                return None

        def _isEmpty(self):
            return len(self.partitionCommandsQueue) == 0

        def _getMaxPriorityInfo(self):
            lengthsList = list(map(len, self.partitionCommandsQueue))
            maxLength = max(lengthsList)
            return maxLength, lengthsList.index(maxLength)

    def __init__(self, tableName: str, tableService: TableService):
        """
        :param tableName: name of table
        :param tableService: instance of :class:`azure.storage.table.TableService` to connect to Azure table storage
        """

        if not AZURE_ALLOWED_TABLE_NAME_PATTERN.match(tableName):
            raise ValueError(f"Invalid table name {tableName}, see: https://docs.microsoft.com/en-us/rest/api/storageservices/Understanding-the-Table-Service-Data-Model")

        self.tableService = tableService
        self.tableName = tableName
        self._partitionQueues = self.PartitionCommandsPriorityQueue()
        self._contextManager = functools.partial(self.tableService.batch, self.tableName)

        if not self.exists():
            self.tableService.create_table(self.tableName)

    def insertOrReplaceEntity(self, entity: Union[Dict, Entity]):
        """
        Lazy wrapper method for :func:`azure.storage.table.TableService.insert_or_replace_entity`
        :param entity:
        """
        partitionKey = entity["PartitionKey"]
        executionCommand = functools.partial(self._insertOrReplaceEntityViaBatch, entity)
        self._partitionQueues.addCommand(partitionKey, executionCommand)

    def insertEntity(self, entity: Union[Dict, Entity]):
        """
        Lazy wrapper method for :func:`azure.storage.table.TableService.insert_entity`
        :param entity:
        """
        partitionKey = entity["PartitionKey"]
        executionCommand = functools.partial(self._insertEntityViaBatch, entity)
        self._partitionQueues.addCommand(partitionKey, executionCommand)

    def getEntity(self, partitionKey: str, rowKey: str):
        """
        Wraps :func:`azure.storage.table.TableService.get_entity`
        :param partitionKey:
        :param rowKey:
        :return:
        """
        return self.tableService.get_entity(self.tableName, partitionKey, rowKey)

    def commitBlockingUntilEmpty(self, maxBatchSize=AZURE_ALLOWED_TABLE_BATCH_SIZE):
        """
        Commit insertion commands. Commands are executed batch-wise per partition until partition queue is empty in a
        blocking manner.
        :param maxBatchSize: maximal batch size to use for batch insertion, must be less or equal to batch size allowed by Azure
        """

        maxBatchSize = self._validateMaxBatchSize(maxBatchSize)

        while not self._partitionQueues.isEmpty():
            commands = self._partitionQueues.pop()
            commands.execute(self._contextManager, maxBatchSize)

    def commitNonBlockingCurrentQueueState(self, maxBatchSize=AZURE_ALLOWED_TABLE_BATCH_SIZE):
        """
        Commit insertion commands. Empties the current PartitionCommandsQueue in a non blocking way.
        Commands are executed batch-wise per partition.
        :param maxBatchSize: maximal batch size to use for batch insertion, must be less or equal to batch size allowed by Azure
        """

        maxBatchSize = self._validateMaxBatchSize(maxBatchSize)

        def commit():
            commandsList = self._partitionQueues.popAll()
            for commands in commandsList:
                commands.execute(self._contextManager, maxBatchSize)

        thread = threading.Thread(target=commit, daemon=False)
        thread.start()

    def commitBlockingLargestPartitionFromQueue(self, maxBatchSize=AZURE_ALLOWED_TABLE_BATCH_SIZE, minLength=None):
        """
        Commits in a blocking way the largest partition from PartitionCommandsQueue
        :param maxBatchSize: maximal batch size to use for batch insertion, must be less or equal to batch size allowed by Azure
        :param minLength: minimal size of largest partition. If not None, pop and commit only if minLength is reached.
        :return:
        """
        maxBatchSize = self._validateMaxBatchSize(maxBatchSize)
        commands = self._partitionQueues.pop(minLength)
        if commands is not None:
            commands.execute(self._contextManager, maxBatchSize)

    @staticmethod
    def _validateMaxBatchSize(maxBatchSize):
        if maxBatchSize > AZURE_ALLOWED_TABLE_BATCH_SIZE:
            _log.warning(f"Provided maxBatchSize is larger than allowed size {AZURE_ALLOWED_TABLE_BATCH_SIZE}. Will use maxBatchSize {AZURE_ALLOWED_TABLE_BATCH_SIZE} instead.")
            maxBatchSize = AZURE_ALLOWED_TABLE_BATCH_SIZE
        return maxBatchSize

    def loadTableToDataFrame(self, columns: List[str] = None, rowFilterQuery: str = None, numRecords: int = None):
        """
        Load all rows of table to :class:`~pandas.DataFrame`
        :param rowFilterQuery:
        :param numRecords:
        :param columns: restrict loading to provided columns
        :return: :class:`~pandas.DataFrame`
        """
        if numRecords is None:
            records = list(self.iterRecords(columns))
        else:
            records = []
            for record in self.iterRecords(columns, rowFilterQuery):
                records.append(record)
                if len(records) >= numRecords:
                    break
        return pd.DataFrame(records, columns=columns)

    def iterDataFrameChunks(self, chunkSize: int, columns: List[str] = None, rowFilterQuery: str = None):
        """
        Get a generator of dataframe chunks
        :param rowFilterQuery:
        :param chunkSize:
        :param columns:
        :return:
        """
        records = []
        for record in self.iterRecords(columns, rowFilterQuery):
            records.append(record)
            if len(records) >= chunkSize:
                yield pd.DataFrame(records, columns=columns)
                records = []

    def iterRecords(self, columns: List[str] = None, rowFilterQuery: str = None) -> Generator[Entity, Any, None]:
        """
        Get a generator of table entities
        :param rowFilterQuery:
        :param columns:
        :return:
        """
        columnNamesAsCommaSeparatedString = None
        if columns is not None:
            columnNamesAsCommaSeparatedString = ",".join(columns)
        for record in self.tableService.query_entities(self.tableName, select=columnNamesAsCommaSeparatedString,
                filter=rowFilterQuery):
            yield record

    def insertDataFrameToTable(self, df: pd.DataFrame, partitionKeyGenerator: Callable[[str], str] = None, numRecords: int = None):
        """
        Inserts or replace entities of the table corresponding to rows of the DataFrame, where the index of the dataFrame acts as rowKey.
        Values of object type columns in the dataFrame may have to be serialised via json beforehand.
        :param df: DataFrame to be inserted
        :param partitionKeyGenerator: if None, partitionKeys default to tableName
        :param numRecords: restrict insertion to first numRecords rows, merely for testing
        """
        for (count, (idx, row)) in enumerate(df.iterrows()):
            if numRecords is not None:
                if count >= numRecords:
                    break
            entity = row.to_dict()
            entity["RowKey"] = idx
            entity["PartitionKey"] = self.tableName if partitionKeyGenerator is None else partitionKeyGenerator(idx)
            self.insertOrReplaceEntity(entity)

    def _insertOrReplaceEntityViaBatch(self, entity, batch: TableBatch):
        return batch.insert_or_replace_entity(entity)

    def _insertEntityViaBatch(self, entity, batch: TableBatch):
        return batch.insert_entity(entity)

    def exists(self):
        return self.tableService.exists(self.tableName)


class AzureTableBlobBackend(ABC):
    """
    Abstraction of a blob backend, which allows for convenient setting and getting of values stored in blob storage via a
    reference to the value
    """

    @abstractmethod
    def getValueFromReference(self, valueIdentifier: str):
        pass

    @abstractmethod
    def getValueReference(self, blobNamePrefix: str, partitionKey: str, rowKey: str, valueName: str) -> str:
        pass

    @abstractmethod
    def setValueForReference(self, valueIdentifier: str, value):
        pass


class BlobPerKeyAzureTableBlobBackend(AzureTableBlobBackend, ABC):

    """
    Backend stores serialised values as /tableName/partitionKey/rowKey/valueName.<fileExtension>
    or /tableName/rowKey/valueName.<fileExtension>, if partitionKey equals tableName
    """

    def __init__(self, blockBlobService: BlockBlobService, containerName: str):
        """

        :param blockBlobService: https://docs.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.blockblobservice.blockblobservice?view=azure-python-previous
        """
        self.blockBlobService = blockBlobService
        self.containerName = containerName
        self.containerList = [container.name for container in blockBlobService.list_containers()]
        if containerName not in self.containerList:
            self.blockBlobService.create_container(containerName)
            self.containerList.append(containerName)

    @property
    @abstractmethod
    def fileExtension(self):
        pass

    @abstractmethod
    def _getBlobValue(self, containerName, blobName):
        pass

    @abstractmethod
    def _writeValueToBlob(self, containerName, blobName, value):
        pass

    def getValueFromReference(self, valueIdentifier: str):
        containerName = self._getContainerNameFromIdentifier(valueIdentifier)
        blobName = self._getBlobNameFromIdentifier(valueIdentifier)
        return self._getBlobValue(containerName, blobName)

    def getValueReference(self, blobNamePrefix: str, partitionKey: str, rowKey: str, valueName: str) -> str:
        blobName = self._getBlobNameFromKeys(blobNamePrefix, partitionKey, rowKey, valueName)
        return self.blockBlobService.make_blob_url(self.containerName, blobName)

    def setValueForReference(self, valueIdentifier: str, value):
        containerName = self._getContainerNameFromIdentifier(valueIdentifier)
        blobName = self._getBlobNameFromIdentifier(valueIdentifier)
        self._writeValueToBlob(containerName, blobName, value)

    def _getBlobNameFromIdentifier(self, valueIdentifier: str):
        return (valueIdentifier.partition(f"{self.blockBlobService.primary_endpoint}/")[2]).partition("/")[2]

    def _getContainerNameFromIdentifier(self, valueIdentifier: str):
        return (valueIdentifier.partition(f"{self.blockBlobService.primary_endpoint}/")[2]).partition("/")[0]

    def _getBlobNameFromKeys(self, tableName: str, partitionKey: str, rowKey: str, valueName: str):
        identifierList = [tableName]
        if tableName != partitionKey:
            identifierList.append(partitionKey)
        identifierList.extend([rowKey, valueName])
        return "/".join(identifierList) + self.fileExtension


class TextDumpAzureTableBlobBackend(BlobPerKeyAzureTableBlobBackend):
    """
   Backend stores values as txt files in the structure /tableName/partitionKey/rowKey/valueName
   """

    @property
    def fileExtension(self):
        return ""

    def _getBlobValue(self, containerName, blobName):
        return self.blockBlobService.get_blob_to_text(containerName, blobName).content

    def _writeValueToBlob(self, containerName, blobName, value):
        self.blockBlobService.create_blob_from_text(containerName, blobName, value)


class JsonAzureTableBlobBackend(BlobPerKeyAzureTableBlobBackend):
    """
    Backend stores values as json files in the structure /tableName/partitionKey/rowKey/valueName.json
    """

    @property
    def fileExtension(self):
        return ".json"

    def _getBlobValue(self, containerName, blobName):
        encodedValue = self.blockBlobService.get_blob_to_bytes(containerName, blobName).content
        return self._decodeBytesToValue(encodedValue)

    def _writeValueToBlob(self, containerName, blobName, value):
        encodedValue = self._encodeValueToBytes(value)
        self.blockBlobService.create_blob_from_bytes(containerName, blobName, encodedValue)

    @staticmethod
    def _encodeValueToBytes(value):
        return str.encode(json.dumps(value))

    @staticmethod
    def _decodeBytesToValue(_bytes):
        return json.loads(_bytes.decode())


class PickleAzureTableBlobBackend(JsonAzureTableBlobBackend):
    """
    Backend stores values as pickle files in the structure /tableName/partitionKey/rowKey/valueName.pickle
    """

    @property
    def fileExtension(self):
        return ".pickle"

    @staticmethod
    def _encodeValueToBytes(value):
        return pickle.dumps(value)

    @staticmethod
    def _decodeBytesToValue(_bytes):
        return pickle.loads(_bytes)


class BlobBackedProperty:
    """
    Abstraction for a table entity property, whose value is backed in Blob storage
    """
    def __init__(self, propertyName: str, blobBackend: AzureTableBlobBackend, max_workers=None):
        """

        :param propertyName: name of property in table storage
        :param blobBackend: actual backend to use for storage
        :param max_workers: maximal number of workers to load data from blob storage
        """
        self.blobBackend = blobBackend
        self.propertyName = propertyName
        self.max_workers = max_workers

    def loadEntityValue(self, entity: Union[Dict, Entity]):
        if self.propertyName in entity.keys():
            entity[self.propertyName] = self.blobBackend.getValueFromReference(entity[self.propertyName])

    def loadValuesToDataFrameColumn(self, df: pd.DataFrame):
        if self.propertyName in df.columns:
            df[self.propertyName] = self._loadValuesInSeries(df[self.propertyName])

    def _loadValuesInSeries(self, _series: pd.Series):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            _series = list(executor.map(self.blobBackend.getValueFromReference, _series))
        return _series

    def writeEntityPropertyToBlob(self, prefix, entity):
        if self.propertyName in entity.keys():
            valueIdentifier = self.blobBackend.getValueReference(prefix, entity["PartitionKey"], entity["RowKey"], self.propertyName)
            value = entity[self.propertyName]
            self.blobBackend.setValueForReference(valueIdentifier, value)
            entity[self.propertyName] = valueIdentifier


class BlobBackedAzureLazyCommitTable(AzureLazyBatchCommitTable):

    """
    Wrapper of an Azure table, which allow for convenient insertion via lazy batch execution per partition.
    Uses a priority queue to manage order of partitions to be committed.
    Can be equipped with blob backed properties, to allow storage, which do not match table storage model, via reference to a blob.
    To execute insertions, call :func:`LazyBatchCommitTable.commit`
    """

    def __init__(self, tableName, tableService: TableService, blobBackedProperties: Sequence[BlobBackedProperty] = ()):
        """

        :param tableName: name of table
        :param tableService: instance of :class:`azure.storage.table.TableService` to connect to Azure table storage
        :param blobBackedProperties:
        """
        self.blobBackedProperties = blobBackedProperties

        super().__init__(tableName, tableService)

    def getEntity(self, partitionKey: str, rowKey: str):
        entity = super().getEntity(partitionKey, rowKey)
        self._loadEntityValues(entity)
        return entity

    def iterRecords(self, columns: List[str] = None, rowFilterQuery: str = None):
        for entity in super().iterRecords(columns, rowFilterQuery):
            self._loadEntityValues(entity)
            yield entity

    def _insertOrReplaceEntityViaBatch(self, entity, batch: TableBatch):
        self._writeEntityPropertiesToBlob(entity)
        return batch.insert_or_replace_entity(entity)

    def _insertEntityViaBatch(self, entity, batch: TableBatch):
        self._writeEntityPropertiesToBlob(entity)
        return batch.insert_entity(entity)

    def _writeEntityPropertiesToBlob(self, entity):
        for blobBackedProperty in self.blobBackedProperties:
            blobBackedProperty.writeEntityPropertyToBlob(prefix=self.tableName, entity=entity)

    def _loadEntityValues(self, entity):
        for blobBackedProperty in self.blobBackedProperties:
            blobBackedProperty.loadEntityValue(entity)


class AzureTablePersistentKeyValueCache(PersistentKeyValueCache):
    """
    PersistentKeyValueCache using Azure Table Storage, see https://docs.microsoft.com/en-gb/azure/storage/tables/
    """
    CACHE_VALUE_IDENTIFIER = "cache_value"

    def __init__(self, tableService: TableService, tableName="cache", partitionKeyGenerator: Callable[[str], str] = None,
            maxBatchSize=100, minSizeForPeriodicCommit: Optional[int] = 100, deferredCommitDelaySecs=1.0, inMemory=False,
            blobBackend: AzureTableBlobBackend = None, max_workers: int = None):
        """


        :param tableService: https://docs.microsoft.com/en-us/python/api/azure-cosmosdb-table/azure.cosmosdb.table.tableservice.tableservice?view=azure-python
        :param tableName: name of table, needs to match restrictions for Azure storage resources, see https://docs.microsoft.com/en-gb/azure/azure-resource-manager/management/resource-name-rules
        :param partitionKeyGenerator: callable to generate a partitionKey from provided string, if None partitionKey in requests defaults to tableName
        :param maxBatchSize: maximal batch size for each commit.
        :param deferredCommitDelaySecs: the time frame during which no new data must be added for a pending transaction to be committed
        :param minSizeForPeriodicCommit: minimal size of a batch to be committed in a periodic thread.
                                         If None, commits are only executed in a deferred manner, i.e. commit only if there is no update for deferredCommitDelaySecs
        :param inMemory: boolean flag, to indicate, if table should be loaded in memory at construction
        :param blobBackend: if not None, blob storage will be used to store actual value and cache_value in table only contains a reference
        :param max_workers: maximal number of workers to load data from blob backend
        """

        self._deferredCommitDelaySecs = deferredCommitDelaySecs
        self._partitionKeyGenerator = partitionKeyGenerator

        if blobBackend is None:
            self._batchCommitTable = AzureLazyBatchCommitTable(tableName, tableService)
        else:
            self._batchCommitTable = BlobBackedAzureLazyCommitTable(tableName, tableService, blobBackedProperties=(BlobBackedProperty(self.CACHE_VALUE_IDENTIFIER, blobBackend, max_workers),))

        self._minSizeForPeriodicCommit = minSizeForPeriodicCommit
        self._maxBatchSize = maxBatchSize
        self._updateHook = PeriodicUpdateHook(deferredCommitDelaySecs, noUpdateFn=self._commit, periodicFn=self._periodicallyCommit)

        self._inMemoryDf = None

        if inMemory:
            df = self._batchCommitTable.loadTableToDataFrame(columns=['RowKey', self.CACHE_VALUE_IDENTIFIER]).set_index("RowKey")
            _log.info(f"Loaded {len(df)} entries of table {tableName} in memory")
            self._inMemoryDf = df

    def set(self, key, value):
        keyAsString = str(key)
        partitionKey = self._getPartitionKeyForRowKey(keyAsString)
        entity = {'PartitionKey': partitionKey, 'RowKey': keyAsString, self.CACHE_VALUE_IDENTIFIER: value}
        self._batchCommitTable.insertOrReplaceEntity(entity)
        self._updateHook.handleUpdate()

        if self._inMemoryDf is not None:
            self._inMemoryDf.loc[keyAsString] = [value]

    def get(self, key):
        keyAsString = str(key)
        value = self._getFromInMemoryDf(keyAsString)
        if value is None:
            value = self._getFromTable(keyAsString)
        return value

    def _getFromTable(self, key: str):
        try:
            partitionKey = self._getPartitionKeyForRowKey(key)
            value = self._batchCommitTable.getEntity(partitionKey, key)[self.CACHE_VALUE_IDENTIFIER]
            return value
        except Exception as e:
            _log.debug(f"Unable to load value for row_key {key}: {e}")
            return None

    def _getFromInMemoryDf(self, key):
        if self._inMemoryDf is None:
            return None
        try:
            return self._inMemoryDf[self.CACHE_VALUE_IDENTIFIER][str(key)]
        except Exception as e:
            _log.debug(f"Unable to load value for key {str(key)} from in-memory dataframe: {e}")
            return None

    def _getPartitionKeyForRowKey(self, key: str):
        return self._batchCommitTable.tableName if self._partitionKeyGenerator is None else self._partitionKeyGenerator(key)

    def _commit(self):
        self._batchCommitTable.commitNonBlockingCurrentQueueState(self._maxBatchSize)

    def _periodicallyCommit(self):
        self._batchCommitTable.commitBlockingLargestPartitionFromQueue(self._maxBatchSize, self._minSizeForPeriodicCommit)
