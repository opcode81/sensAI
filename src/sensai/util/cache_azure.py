import collections
import functools
import pickle
from abc import ABC, abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Callable, Dict, Union, Any, List, Sequence
import json
import logging
import re
import threading
import time

from azure.storage.table import TableService, TableBatch, Entity
from azure.storage.blob import BlockBlobService
import pandas as pd

from .cache import PersistentKeyValueCache


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

        def addCommand(self, partitionKey, command: Callable[[TableBatch], Any]):
            if partitionKey not in self.partitionKey2Commands:
                commands = self.PartitionCommands(partitionKey)
                self.partitionCommandsQueue.append(commands)
                self.partitionKey2Commands[partitionKey] = commands
            self.partitionKey2Commands[partitionKey].append(command)

        def pop(self):
            maxIndex = self._getIndexOfMaxPriority()
            q = self.partitionCommandsQueue.pop(maxIndex)
            del self.partitionKey2Commands[q.partitionKey]
            return q

        def isEmpty(self):
            return len(self.partitionCommandsQueue) == 0

        def _getIndexOfMaxPriority(self):
            lengthsList = list(map(len, self.partitionCommandsQueue))
            return lengthsList.index(max(lengthsList))

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

    def commit(self, maxBatchSize=AZURE_ALLOWED_TABLE_BATCH_SIZE, minSecsBetweenPartitionCommits: Union[int, float] = None):
        """
        Commit insertion commands. Commands are executed batch-wise per partition
        :param maxBatchSize: maximal batch size to use for batch insertion, must be less or equal to batch size allowed by Azure
        :param minSecsBetweenPartitionCommits: min seconds to wait between two partitions to be committed.
               In case the number of commands to be committed is dynamic, increasing this can lead to larger batches and
               fewer requests
        """

        if maxBatchSize > AZURE_ALLOWED_TABLE_BATCH_SIZE:
            _log.warning(f"Provided maxBatchSize is larger than allowed size {AZURE_ALLOWED_TABLE_BATCH_SIZE}. Will use maxBatchSize {AZURE_ALLOWED_TABLE_BATCH_SIZE} instead.")

        while not self._partitionQueues.isEmpty():
            if minSecsBetweenPartitionCommits is not None:
                time.sleep(minSecsBetweenPartitionCommits)
            commands = self._partitionQueues.pop()
            commands.execute(self._contextManager, maxBatchSize)

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

    def iterRecords(self, columns: List[str] = None, rowFilterQuery: str = None):
        """

        :param numResults:
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
    Backend stores serialised values in the structure /tableName/partitionKey/rowKey/valueName.<fileExtension>
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
    def __init__(self, propertyName: str, blobBackend: AzureTableBlobBackend, max_workers=None):
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
        if self.max_workers is None:
            return [self.blobBackend.getValueFromReference(reference) for reference in _series]
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
    Can be equipped with a blob storage backend, to allow storage of entity properties, which are too large for table storage.
    To execute insertions, call :func:`LazyBatchCommitTable.commit`
    """

    def __init__(self, tableName, tableService: TableService, blobBackedProperties: Sequence[BlobBackedProperty] = ()):
        """

        :param tableName:
        :param tableService:
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
            maxBatchSize=100, deferredCommitDelaySecs=10.0, inMemory=False, blobBackend: AzureTableBlobBackend = None):
        """

        :param tableService: https://docs.microsoft.com/en-us/python/api/azure-cosmosdb-table/azure.cosmosdb.table.tableservice.tableservice?view=azure-python
        :param jsonSerialisedValues: boolean flag, to indicate if values must be serialised via json
        :param tableName: name of table, needs to match restrictions for Azure storage resources, see https://docs.microsoft.com/en-gb/azure/azure-resource-manager/management/resource-name-rules
        :param partitionKeyGenerator: callable to generate a partitionKey from provided string, if None partitionKey in requests defaults to tableName
        :param inMemory: boolean flag, to indicate, if table should be loaded in memory at construction
        :param blobBackend: if not None, blob storage will be used to store actual value and filed cache_value in table only contains a reference
        """

        self._deferredCommitDelaySecs = deferredCommitDelaySecs
        self._partitionKeyGenerator = partitionKeyGenerator
        self._batchCommitTable = self._getBatchCommitTable(tableName, tableService, blobBackend)
        self._maxBatchSize = maxBatchSize
        self._commitThread = None
        self._commitThreadSemaphore = threading.Semaphore()
        self._numEntriesToBeCommitted = 0
        self._lastUpdateTime = None
        self._inMemoryDf = None
        self._lastUpdateTime = time.time()

        if inMemory:
            df = self._batchCommitTable.loadTableToDataFrame(columns=['RowKey', self.CACHE_VALUE_IDENTIFIER]).set_index("RowKey")
            _log.info(f"Loaded {len(df)} entries of table {tableName} in memory")
            self._inMemoryDf = df

    def _getBatchCommitTable(self, tableName, tableService, blobBackend: AzureTableBlobBackend):
        if blobBackend is None:
            return AzureLazyBatchCommitTable(tableName, tableService)
        return BlobBackedAzureLazyCommitTable(tableName, tableService, blobBackedProperties=(BlobBackedProperty(self.CACHE_VALUE_IDENTIFIER, blobBackend),))

    def set(self, key, value):
        keyAsString = str(key)
        partitionKey = self._getPartitionKeyForRowKey(keyAsString)
        entity = {'PartitionKey': partitionKey, 'RowKey': keyAsString, self.CACHE_VALUE_IDENTIFIER: value}
        self._batchCommitTable.insertOrReplaceEntity(entity)
        self._numEntriesToBeCommitted += 1
        self._commitDeferred()

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

    def _commitDeferred(self):

        def doCommit():
            self._batchCommitTable.commit(self._maxBatchSize, self._deferredCommitDelaySecs)

        if self._commitThread is None or not self._commitThread.is_alive():
            self._commitThreadSemaphore.acquire()
            if self._commitThread is None or not self._commitThread.is_alive():
                self._commitThread = threading.Thread(target=doCommit, daemon=False)
                self._commitThread.start()
            self._commitThreadSemaphore.release()