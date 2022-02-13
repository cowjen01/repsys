import React, { useState } from 'react';
import { Grid, Typography, Box, LinearProgress } from '@mui/material';

import ItemsDescriptionPanel from './ItemsDescriptionPanel';
import UserDescriptionPanel from './UserDescriptionPanel';
import ErrorAlert from '../ErrorAlert';
import { useGetDatasetQuery } from '../../api';
import ItemsEmbeddings from './ItemsEmbeddings';
import UsersEmbeddings from './UsersEmbeddings';

function DatasetEvaluation() {
  const [selectedItems, setSelectedItems] = useState([]);
  const [selectedUsers, setSelectedUsers] = useState([]);

  const dataset = useGetDatasetQuery();

  if (dataset.isLoading) {
    return <LinearProgress />;
  }

  if (dataset.isError) {
    return <ErrorAlert error={dataset.error} />;
  }

  const { attributes } = dataset.data;

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Box pl={1}>
          <Typography component="div" variant="h6">
            Item Embeddings
          </Typography>
          <Typography variant="subtitle1" gutterBottom>
            Exploration of the item embeddings created from the training split
          </Typography>
        </Box>
        <Grid container spacing={2} sx={{ height: 560 }}>
          <Grid item xs={8} sx={{ height: '100%' }}>
            <ItemsEmbeddings attributes={attributes} onSelect={(ids) => setSelectedItems(ids)} />
          </Grid>
          <Grid item xs={4} sx={{ height: '100%' }}>
            <ItemsDescriptionPanel attributes={attributes} items={selectedItems} />
          </Grid>
        </Grid>
      </Grid>
      <Grid item xs={12}>
        <Box pl={1}>
          <Typography component="div" variant="h6">
            User Embeddings
          </Typography>
          <Typography variant="subtitle1" gutterBottom>
            Exploration of the user embeddings regulated by a minimum number of interactions
          </Typography>
        </Box>
        <Grid container spacing={2} sx={{ height: 560 }}>
          <Grid item xs={8} sx={{ height: '100%' }}>
            <UsersEmbeddings attributes={attributes} onSelect={(ids) => setSelectedUsers(ids)} />
          </Grid>
          <Grid item xs={4} sx={{ height: '100%' }}>
            <UserDescriptionPanel users={selectedUsers} />
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
}

export default DatasetEvaluation;
