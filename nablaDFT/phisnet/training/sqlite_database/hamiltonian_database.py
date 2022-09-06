# coding:utf-8
import math
import multiprocessing
import os

import apsw  # way faster than sqlite3
import numpy as np

'''
This is a class to store large amounts of ab initio reference data
for training a neural network in a SQLite database

Data structure:
input data:
 Z (N)    (int)   nuclear charges
 R (N, 3) (float) Cartesian coordinates in bohr
output data:
 E ()     (float) energy in Eh
 F (N, 3) (float) forces in Eh/bohr
 H (Norb, Norb)   full hamiltonian in atomic units
 S (Norb, Norb)   overlap matrix in atomic units
 C (Norb, Norb)   core hamiltonian in atomic units
'''


class HamiltonianDatabase:
    def __init__(self, filename, flags=apsw.SQLITE_OPEN_READONLY):
        self.db = filename
        self.connections = {}  # allow multiple connections (needed for multi-threading)
        self._open(flags=flags)  # creates the database if it doesn't exist yet

    def __len__(self):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        return cursor.execute('''SELECT * FROM metadata WHERE id=0''').fetchone()[-1]

    def __getitem__(self, idx):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()

        if type(idx) == list:  # for batched data retrieval
            data = cursor.execute('''SELECT * FROM data WHERE id IN (''' + str(idx)[1:-1] + ')').fetchall()
            return [self._unpack_data_tuple(i) for i in data]
        else:
            data = cursor.execute('''SELECT * FROM data WHERE id=''' + str(idx)).fetchone()
            return self._unpack_data_tuple(data)

    def _unpack_data_tuple(self, data):

        N = len(data[2]) // 4 // 3  # a single float32 is 4 bytes, we have 3 in data[1] (positions)
        R = self._deblob(data[2], dtype=np.float32, shape=(N, 3))
        Z = self._deblob(data[1], dtype=np.int32, shape=(N))
        E = np.array([0.0 if data[3] is None else data[3]], dtype=np.float32)
        F = self._deblob(data[4], dtype=np.float32, shape=(N, 3))
        Norb = int(math.sqrt(len(data[5]) // 4))  # a single float32 is 4 bytes, we have Norb**2 of them
        H = self._deblob(data[5], dtype=np.float32, shape=(Norb, Norb))
        S = self._deblob(data[6], dtype=np.float32, shape=(Norb, Norb))
        C = self._deblob(data[7], dtype=np.float32, shape=(Norb, Norb))
        return Z, R, E, F, H, S, C

    def add_data(self, Z, R, E, F, H, S, C, flags=apsw.SQLITE_OPEN_READWRITE, transaction=True):

        # check that no NaN values are added
        if self._any_is_nan(Z, R, E, F, H, S, C):
            print("encountered NaN, data is not added")
            return

        cursor = self._get_connection(flags=flags).cursor()

        # update data
        if transaction:
            # begin exclusive transaction (locks db) which is necessary
            # if database is accessed from multiple programs at once (default for safety)
            cursor.execute('''BEGIN EXCLUSIVE''')
        try:
            length = cursor.execute('''SELECT * FROM metadata WHERE id=0''').fetchone()[-1]
            cursor.execute('''INSERT INTO data (id, Z, R, E, F, H, S, C) VALUES (?,?,?,?,?,?,?,?)''',
                           (None if length > 0 else 0,  # autoincrementing ID
                            self._blob(Z), self._blob(R), None if E is None else float(E),
                            self._blob(F), self._blob(H), self._blob(S), self._blob(C)))
            # update metadata
            cursor.execute('''INSERT OR REPLACE INTO metadata VALUES (?,?)''', (0, length + 1))

            if transaction:
                cursor.execute('''COMMIT''')  # end transaction

        except Exception as exc:
            if transaction:
                cursor.execute('''ROLLBACK''')
            raise exc

    def add_orbitals(self, Z, orbitals, flags=apsw.SQLITE_OPEN_READWRITE):
        cursor = self._get_connection(flags=flags).cursor()
        cursor.execute('''INSERT OR REPLACE INTO basisset (Z, orbitals) VALUES (?,?)''',
                       (int(Z), self._blob(orbitals)))

    def get_orbitals(self, Z):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        data = cursor.execute('''SELECT * FROM basisset WHERE Z=''' + str(Z)).fetchone()
        Norb = len(data[1]) // 4  # each entry is 4 bytes
        return self._deblob(data[1], dtype=np.int32, shape=(Norb,))

    def _any_is_nan(self, *vals):
        nan = False

        for val in vals:
            if val is None:
                continue
            nan = nan or np.any(np.isnan(val))
        return nan

    def _blob(self, array):
        """Convert numpy array to blob/buffer object."""
        if array is None:
            return None

        if array.dtype == np.float64:
            array = array.astype(np.float32)
        if array.dtype == np.int64:
            array = array.astype(np.int32)
        if not np.little_endian:
            array = array.byteswap()

        return memoryview(np.ascontiguousarray(array))

    def _deblob(self, buf, dtype=np.float32, shape=None):
        """Convert blob/buffer object to numpy array."""
        if buf is None:
            return np.zeros(shape)

        array = np.frombuffer(buf, dtype)
        if not np.little_endian:
            array = array.byteswap()

        array.shape = shape
        return array

    def _open(self, flags=apsw.SQLITE_OPEN_READONLY):

        newdb = not os.path.isfile(self.db)
        cursor = self._get_connection(flags=flags).cursor()

        if newdb:
            # create table to store data
            cursor.execute('''CREATE TABLE IF NOT EXISTS data
                (id INTEGER NOT NULL PRIMARY KEY,
                 Z BLOB,
                 R BLOB,
                 E FLOAT,
                 F BLOB,
                 H BLOB,
                 S BLOB,
                 C BLOB
                )''')

            # create table to store things that are constant for the whole dataset
            cursor.execute('''CREATE TABLE IF NOT EXISTS nuclear_charges
                (id INTEGER NOT NULL PRIMARY KEY, N INTEGER, Z BLOB)''')
            cursor.execute('''INSERT OR IGNORE INTO nuclear_charges (id, N, Z) VALUES (?,?,?)''',
                           (0, 1, self._blob(np.array([0]))))
            self.N = len(self.Z)

            # create table to store the basis set convention
            cursor.execute('''CREATE TABLE IF NOT EXISTS basisset
                (Z INTEGER NOT NULL PRIMARY KEY, orbitals BLOB)''')

            # create table to store metadata (information about the number of entries)
            cursor.execute('''CREATE TABLE IF NOT EXISTS metadata
                (id INTEGER PRIMARY KEY, N INTEGER)''')
            cursor.execute('''INSERT OR IGNORE INTO metadata (id, N) VALUES (?,?)''', (0, 0))  # num_data

    def _get_connection(self, flags=apsw.SQLITE_OPEN_READONLY):
        '''
        This allows multiple processes to access the database at once,
        every process must have its own connection
        '''
        key = multiprocessing.current_process().name

        if key not in self.connections.keys():
            self.connections[key] = apsw.Connection(self.db, flags=flags)
            self.connections[key].setbusytimeout(300000)  # 5 minute timeout

        return self.connections[key]

    def add_Z(self, Z, flags=apsw.SQLITE_OPEN_READWRITE):
        cursor = self._get_connection(flags=flags).cursor()
        self.N = len(Z)
        cursor.execute('''INSERT OR REPLACE INTO nuclear_charges (id, N, Z) VALUES (?,?,?)''',
                       (0, self.N, self._blob(Z)))

    @property
    def Z(self):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        data = cursor.execute('''SELECT * FROM nuclear_charges WHERE id=0''').fetchone()
        N = data[1]
        return self._deblob(data[2], dtype=np.int32, shape=(N,))
