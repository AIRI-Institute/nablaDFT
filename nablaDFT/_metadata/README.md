### Datasources metadata

All available datasources must have described metadata in json format:  
```json
{
    "name": "datasource name, used for download and instantiation",
    "desc": "datasource description",
    "doi": "paper doi with dataset description if available",
    "metadata": "must contain information about types of computational methods and software used for datasource generation",
    "keys_map": "must contain mapping between database keys and sample keys",
    "data_dtypes": "must contain mapping between database keys and data types",
    "data_shapes": "optional, contains shape of data for selection of database columns"
}
```

> Note: for multi-table database `keys_map` datasource key must contain table name and column name in format `table_name.column_name`.