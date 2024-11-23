### Datasources metadata

Available datasources must have described metadata in json format:  
```json
{
    "desc": "datasource description",
    "metadata": "must contain information about types of computational methods and software used for datasource generation",
    "columns": "describes columns from datasource to retrieve",
    "_keys_map": "must contain mapping between database keys and sample keys",
    "_data_dtypes": "must contain mapping between database keys and data types",
    "_data_shapes": "optional, contains shape of data for non-flat data"
}
```