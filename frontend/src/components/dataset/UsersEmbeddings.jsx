import React, { useState } from 'react';
import pt from 'prop-types';
import { Box, Grid, Paper } from '@mui/material';

import ErrorAlert from '../ErrorAlert';
import EmbeddingsPlot from './EmbeddingsPlot';
import { useGetUserEmbeddingsQuery, useSearchUsersMutation } from '../../api';
import { PlotLoader } from '../loaders';
import AttributesSelector from './AttributesSelector';
import UsersDescription from './UsersDescription';

function UsersEmbeddings({ attributes, split, displayFilters }) {
  const [filterResetIndex, setFilterResetIndex] = useState(0);
  const [plotResetIndex, setPlotResetIndex] = useState(0);
  const [selectedUsers, setSelectedUsers] = useState([]);
  const [isPlotLoading, setIsPlotLoading] = useState(false);
  const embeddings = useGetUserEmbeddingsQuery();
  const [searchUsers, users] = useSearchUsersMutation();

  if (embeddings.isError) {
    return <ErrorAlert error={embeddings.error} />;
  }

  if (users.isError) {
    return <ErrorAlert error={users.error} />;
  }

  const handleFilterApply = (query) => {
    searchUsers({
      split,
      query,
    });
  };

  const handlePlotUnselect = () => {
    setFilterResetIndex(filterResetIndex + 1);
    setSelectedUsers([]);
  };

  const handlePlotSelect = (ids) => {
    setFilterResetIndex(filterResetIndex + 1);
    setSelectedUsers(ids);
  };

  const handleFilterChange = () => {
    setPlotResetIndex(plotResetIndex + 1);
    setSelectedUsers([]);
  };

  const isLoading = embeddings.isLoading || users.isLoading || isPlotLoading;

  return (
    <Grid container spacing={2}>
      {displayFilters && (
        <Grid item xs={12}>
          <AttributesSelector
            onChange={handleFilterChange}
            resetIndex={filterResetIndex}
            disabled={isLoading}
            attributes={attributes}
            onFilterApply={handleFilterApply}
            displayThreshold
          />
        </Grid>
      )}
      <Grid item xs={12}>
        <Grid container spacing={2} sx={{ height: 500 }}>
          <Grid item xs={8}>
            <Box position="relative">
              {isLoading && <PlotLoader />}
              <Paper sx={{ p: 2 }}>
                <EmbeddingsPlot
                  resetIndex={plotResetIndex}
                  onUnselect={handlePlotUnselect}
                  embeddings={embeddings.data}
                  onSelect={handlePlotSelect}
                  selectedIds={users.data}
                  isLoading={isLoading}
                  onComputeStarted={() => setIsPlotLoading(true)}
                  onComputeFinished={() => setIsPlotLoading(false)}
                  markerSize={3}
                  markerOpacity={0.3}
                />
              </Paper>
            </Box>
          </Grid>
          <Grid item xs={4} sx={{ height: '100%' }}>
            {selectedUsers.length > 0 ? (
              <Paper sx={{ p: 2, height: '100%', overflow: 'auto' }}>
                <UsersDescription attributes={attributes} users={selectedUsers} />
              </Paper>
            ) : (
              <Paper
                sx={{
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'text.secondary',
                }}
              >
                Select a subset of the space.
              </Paper>
            )}
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
}

UsersEmbeddings.defaultProps = {
  split: 'train',
  displayFilters: true,
};

UsersEmbeddings.propTypes = {
  // eslint-disable-next-line react/forbid-prop-types
  attributes: pt.any.isRequired,
  split: pt.string,
  displayFilters: pt.bool,
};

export default UsersEmbeddings;
