import React, { useState } from 'react';
import pt from 'prop-types';
import { Box, Grid, Paper } from '@mui/material';

import ErrorAlert from '../ErrorAlert';
import EmbeddingsPlot from './EmbeddingsPlot';
import { useGetItemsEmbeddingsQuery, useSearchItemsMutation } from '../../api';
import { PlotLoader } from '../loaders';
import AttributesSelector from './AttributesSelector';
import ItemsDescription from './ItemsDescription';

function ItemsEmbeddings({ attributes }) {
  const [filterResetIndex, setFilterResetIndex] = useState(0);
  const [plotResetIndex, setPlotResetIndex] = useState(0);
  const [selectedItems, setSelectedItems] = useState([]);
  const embeddings = useGetItemsEmbeddingsQuery();
  const [searchItems, items] = useSearchItemsMutation();

  if (embeddings.isError) {
    return <ErrorAlert error={embeddings.error} />;
  }

  if (items.isError) {
    return <ErrorAlert error={items.error} />;
  }

  const handleFilterApply = (query) => {
    searchItems({ query });
  };

  const handlePlotUnselect = () => {
    setFilterResetIndex(filterResetIndex + 1);
    setSelectedItems([]);
  };

  const handlePlotSelect = (ids) => {
    setFilterResetIndex(filterResetIndex + 1);
    setSelectedItems(ids);
  };

  const handleFilterChange = () => {
    setPlotResetIndex(plotResetIndex + 1);
    setSelectedItems([]);
  };

  const isLoading = embeddings.isLoading || items.isLoading;

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <AttributesSelector
          onChange={handleFilterChange}
          resetIndex={filterResetIndex}
          disabled={isLoading}
          attributes={attributes}
          onFilterApply={handleFilterApply}
        />
      </Grid>
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
                  filterResults={items.data}
                />
              </Paper>
            </Box>
          </Grid>
          <Grid item xs={4} sx={{ height: '100%' }}>
            {selectedItems.length > 0 && (
              <Paper sx={{ p: 3, height: '100%', overflow: 'auto' }}>
                <ItemsDescription attributes={attributes} items={selectedItems} />
              </Paper>
            )}
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
}

ItemsEmbeddings.propTypes = {
  // eslint-disable-next-line react/forbid-prop-types
  attributes: pt.any.isRequired,
};

export default ItemsEmbeddings;
