import React from 'react';
import { LinearProgress, Box } from '@mui/material';
import { useParams } from 'react-router-dom';

import ErrorAlert from '../ErrorAlert';
import { useGetDatasetQuery } from '../../api';
import ItemsEmbeddings from '../dataset/ItemsEmbeddings';
import UsersEmbeddings from '../dataset/UsersEmbeddings';

function DatasetWidget() {
  const dataset = useGetDatasetQuery();
  const { dataType } = useParams();

  return (
    <Box sx={{ p: 2 }}>
      {dataset.isLoading && <LinearProgress />}
      {!dataset.isLoading && dataset.isError && <ErrorAlert error={dataset.error} />}
      {!dataset.isLoading && !dataset.isError && dataType === 'items' && (
        <ItemsEmbeddings attributes={dataset.data.attributes} />
      )}
      {!dataset.isLoading && !dataset.isError && dataType === 'users' && (
        <UsersEmbeddings attributes={dataset.data.attributes} />
      )}
    </Box>
  );
}

export default DatasetWidget;
