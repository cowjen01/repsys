import React from 'react';
import pt from 'prop-types';
import { Box } from '@mui/material';

import ErrorAlert from '../ErrorAlert';
import EmbeddingsPlot from './EmbeddingsPlot';
import { useGetItemsEmbeddingsQuery, useSearchItemsByAttributeMutation } from '../../api';
import PlotLoader from '../PlotLoader';

function ItemsEmbeddings({ attributes, onSelect }) {
  const embeddings = useGetItemsEmbeddingsQuery();
  const [searchItems, items] = useSearchItemsByAttributeMutation();

  if (embeddings.isError) {
    return <ErrorAlert error={embeddings.error} />;
  }

  if (items.isError) {
    return <ErrorAlert error={items.error} />;
  }

  const handleFilterApply = (query) => {
    searchItems(query);
  };

  return (
    <Box position="relative">
      {(embeddings.isLoading || items.isLoading) && <PlotLoader />}
      <EmbeddingsPlot
        embeddings={embeddings.data}
        attributes={attributes}
        onSelect={onSelect}
        onFilterApply={handleFilterApply}
        filterResults={items.data}
      />
    </Box>
  );
}

ItemsEmbeddings.propTypes = {
  onSelect: pt.func.isRequired,
  // eslint-disable-next-line react/forbid-prop-types
  attributes: pt.any.isRequired,
};

export default ItemsEmbeddings;
