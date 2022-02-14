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
        <Typography component="div" variant="h6">
          Items Embeddings
        </Typography>
        <Typography variant="subtitle1" marginBottom={1}>
          Filter the items by attributes value or select a cluster to see the description.
        </Typography>
        <ItemsEmbeddings attributes={dataset.data.attributes} />
      </Grid>
      <Grid item xs={12}>
        <Typography component="div" variant="h6">
          Users Embeddings
        </Typography>
        <Typography variant="subtitle1" marginBottom={1}>
          Filter the users by a minimum number of interactions with an attribute or select a cluster
          to see the description.
        </Typography>
        <UsersEmbeddings attributes={dataset.data.attributes} />
      </Grid>
    </Grid>
  );
}

export default DatasetEvaluation;
