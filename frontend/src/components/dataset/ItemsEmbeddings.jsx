import React, { useState } from 'react';
import pt from 'prop-types';
import { Box, Grid, Paper } from '@mui/material';

import ErrorAlert from '../ErrorAlert';
import EmbeddingsPlot from './EmbeddingsPlot';
import { useGetItemEmbeddingsQuery, useSearchItemsMutation } from '../../api';
import { PlotLoader } from '../loaders';
import AttributesSelector from './AttributesSelector';
import ItemsDescription from './ItemsDescription';

function ItemsEmbeddings({ attributes, displayFilters }) {
  const [filterResetIndex, setFilterResetIndex] = useState(0);
  const [plotResetIndex, setPlotResetIndex] = useState(0);
  const [selectedItems, setSelectedItems] = useState([]);
  const [isPlotLoading, setIsPlotLoading] = useState(false);
  const embeddings = useGetItemEmbeddingsQuery();
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

  const isLoading = embeddings.isLoading || items.isLoading || isPlotLoading;

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
                  selectedIds={items.data}
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
            {selectedItems.length > 0 ? (
              <Paper sx={{ p: 2, height: '100%', overflow: 'auto' }}>
                <ItemsDescription attributes={attributes} items={selectedItems} />
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

ItemsEmbeddings.defaultProps = {
  displayFilters: true,
};

ItemsEmbeddings.propTypes = {
  // eslint-disable-next-line react/forbid-prop-types
  attributes: pt.any.isRequired,
  displayFilters: pt.bool,
};

export default ItemsEmbeddings;
