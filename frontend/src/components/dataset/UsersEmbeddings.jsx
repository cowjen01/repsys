import React from 'react';
import pt from 'prop-types';
import { Box } from '@mui/material';

import ErrorAlert from '../ErrorAlert';
import EmbeddingsPlot from './EmbeddingsPlot';
import { useGetUsersEmbeddingsQuery, useSearchUsersByInteractionsMutation } from '../../api';
import PlotLoader from '../PlotLoader';

function UsersEmbeddings({ attributes, onSelect }) {
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
      query,
    });
  };

  return (
    <Box position="relative">
      {(embeddings.isLoading || users.isLoading) && <PlotLoader />}
      <EmbeddingsPlot
        embeddings={embeddings.data}
        attributes={attributes}
        displayThreshold
        onSelect={onSelect}
        onFilterApply={handleFilterApply}
        filterResults={users.data}
      />
    </Box>
  );
}

UsersEmbeddings.propTypes = {
  onSelect: pt.func.isRequired,
  // eslint-disable-next-line react/forbid-prop-types
  attributes: pt.any.isRequired,
};

export default UsersEmbeddings;
