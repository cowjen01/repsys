import React from 'react';
import { Grid, LinearProgress, Typography } from '@mui/material';

import ErrorAlert from '../ErrorAlert';
import { useGetDatasetQuery } from '../../api';
import ItemsEmbeddings from './ItemsEmbeddings';
import UsersEmbeddings from './UsersEmbeddings';

function DatasetEvaluation() {
  const dataset = useGetDatasetQuery();

  if (dataset.isLoading) {
    return <LinearProgress />;
  }

  if (dataset.isError) {
    return <ErrorAlert error={dataset.error} />;
  }

  return (
    <Grid container spacing={4}>
      <Grid item xs={12}>
        <Typography component="div" variant="h6" gutterBottom>
          Items Embeddings
        </Typography>
        <ItemsEmbeddings attributes={dataset.data.attributes} />
      </Grid>
      <Grid item xs={12}>
        <Typography component="div" variant="h6" gutterBottom>
          Users Embeddings
        </Typography>
        <UsersEmbeddings attributes={dataset.data.attributes} />
      </Grid>
    </Grid>
  );
}

export default DatasetEvaluation;
