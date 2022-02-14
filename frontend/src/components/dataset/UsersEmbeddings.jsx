import React, { useState } from 'react';
import pt from 'prop-types';
import { Box, Typography, Grid } from '@mui/material';

import ErrorAlert from '../ErrorAlert';
import { EmbeddingsPlot } from '../plots';
import { useGetUsersEmbeddingsQuery, useSearchUsersByInteractionsMutation } from '../../api';
import PlotLoader from '../PlotLoader';
import AttributeFilter from './AttributeFilter';
import UsersDescription from './UsersDescription';

function UsersEmbeddings({ attributes, split }) {
  const [filterResetIndex, setFilterResetIndex] = useState(0);
  const [plotResetIndex, setPlotResetIndex] = useState(0);
  const [selectedUsers, setSelectedUsers] = useState([]);
  const embeddings = useGetUsersEmbeddingsQuery();
  const [searchUsers, users] = useSearchUsersByInteractionsMutation();

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

  const isLoading = embeddings.isLoading || users.isLoading;

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <AttributeFilter
          onChange={handleFilterChange}
          resetIndex={filterResetIndex}
          disabled={isLoading}
          attributes={attributes}
          onFilterApply={handleFilterApply}
          displayThreshold
        />
      </Grid>
      <Grid item xs={12}>
        <Grid container spacing={2} sx={{ height: 500 }}>
          <Grid item xs={8} sx={{ height: '100%' }}>
            <Box position="relative" height="100%">
              {isLoading && <PlotLoader />}
              <EmbeddingsPlot
                resetIndex={plotResetIndex}
                onUnselect={handlePlotUnselect}
                embeddings={embeddings.data}
                onSelect={handlePlotSelect}
                filterResults={users.data}
              />
            </Box>
          </Grid>
          <Grid item xs={4} sx={{ height: '100%' }}>
            <UsersDescription attributes={attributes} users={selectedUsers} />
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
}

UsersEmbeddings.defaultProps = {
  split: 'train',
};

UsersEmbeddings.propTypes = {
  // eslint-disable-next-line react/forbid-prop-types
  attributes: pt.any.isRequired,
  split: pt.string,
};

export default UsersEmbeddings;
