import React, { useState } from 'react';
import { Container, Grid, Typography, Box } from '@mui/material';

import EmbeddingsPlot from './EmbeddingsPlot';
import ItemDescriptionPanel from './ItemDescriptionPanel';
import UserDescriptionPanel from './UserDescriptionPanel';

const columns = {
  year: {
    dtype: 'number',
    bins: [
      [0, 1983],
      [1984, 2000],
      [2001, 2005],
      [2006, 2020],
    ],
  },
  genres: {
    dtype: 'tags',
    options: ['action', 'drama', 'comedy', 'horror'],
  },
  country: {
    dtype: 'category',
    options: ['CO', 'MK', 'CN', 'FR', 'ID'],
  },
};

function DatasetEvaluation() {
  const [selectedItems, setSelectedItems] = useState([]);
  const [selectedUsers, setSelectedUsers] = useState([]);

  return (
    <Container maxWidth="xl">
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
              <EmbeddingsPlot
                dataType="items"
                columns={columns}
                onSelect={(ids) => setSelectedItems(ids)}
              />
            </Grid>
            <Grid item xs={4} sx={{ height: '100%' }}>
              <ItemDescriptionPanel columns={columns} itemIds={selectedItems} />
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
              <EmbeddingsPlot
                dataType="users"
                columns={columns}
                onSelect={(ids) => setSelectedUsers(ids)}
              />
            </Grid>
            <Grid item xs={4} sx={{ height: '100%' }}>
              <UserDescriptionPanel userIds={selectedUsers} />
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Container>
  );
}

export default DatasetEvaluation;
