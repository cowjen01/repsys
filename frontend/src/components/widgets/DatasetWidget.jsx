import React from 'react';
import { LinearProgress } from '@mui/material';
import { useParams } from 'react-router-dom';

import ErrorAlert from '../ErrorAlert';
import { useGetDatasetQuery } from '../../api';
import ItemsEmbeddings from '../dataset/ItemsEmbeddings';
import UsersEmbeddings from '../dataset/UsersEmbeddings';

function DatasetWidget() {
  const dataset = useGetDatasetQuery();
  const { dataType } = useParams();

  if (dataset.isLoading) {
    return <LinearProgress />;
  }

  if (dataset.isError) {
    return <ErrorAlert error={dataset.error} />;
  }

  if (dataType === 'items') {
    return <ItemsEmbeddings displayFilters={false} attributes={dataset.data.attributes} />;
  }

  if (dataType === 'users') {
    return <UsersEmbeddings displayFilters={false} attributes={dataset.data.attributes} />;
  }

  return null;
}

export default DatasetWidget;
