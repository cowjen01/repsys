import React, { useState, useMemo, useEffect } from 'react';
import pt from 'prop-types';
import { Paper, Grid, Box, Stack } from '@mui/material';

import UsersDescription from '../dataset/UsersDescription';
import ItemsDescription from '../dataset/ItemsDescription';
import {
  useGetUserMetricsByModelQuery,
  useGetUserEmbeddingsQuery,
  useGetItemEmbeddingsQuery,
  useGetItemMetricsByModelQuery,
} from '../../api';
import { CategoryFilter } from '../filters';
import EmbeddingsPlot from '../dataset/EmbeddingsPlot';
import { PlotLoader } from '../loaders';

function MetricsEmbeddings({ metricsType, itemAttributes, evaluatedModels }) {
  const [selectedData, setSelectedData] = useState([]);
  const [selectedMarkerSize, setSelectedMarkerSize] = useState(3);
  const [selectedModel, setSelectedModel] = useState(evaluatedModels[0]);
  const [selectedMetric, setSelectedMetric] = useState('');
  const [plotResetIndex, setPlotResetIndex] = useState(0);
  const [selectedColorScale, setSelectedColorScale] = useState('Picnic');

  const userEmbeddings = useGetUserEmbeddingsQuery('validation', {
    skip: metricsType !== 'user',
  });
  const itemEmbeddings = useGetItemEmbeddingsQuery('train', {
    skip: metricsType !== 'item',
  });
  const embeddings = metricsType === 'user' ? userEmbeddings : itemEmbeddings;

  const userMetrics = useGetUserMetricsByModelQuery(selectedModel, {
    skip: metricsType !== 'user',
  });
  const itemMetrics = useGetItemMetricsByModelQuery(selectedModel, {
    skip: metricsType !== 'item',
  });
  const metrics = metricsType === 'user' ? userMetrics : itemMetrics;

  const handleModelChange = (newValue) => {
    setSelectedModel(newValue);
    setSelectedMetric('');
    setPlotResetIndex(plotResetIndex + 1);
    setSelectedData([]);
  };

  const handleMetricChange = (newValue) => {
    setSelectedMetric(newValue);
    setPlotResetIndex(plotResetIndex + 1);
    setSelectedData([]);
  };

  const handlePlotUnselect = () => {
    setSelectedData([]);
  };

  const handlePlotSelect = (ids) => {
    setSelectedData(ids);
  };

  const embeddingsColor = useMemo(() => {
    if (metrics.data) {
      return metrics.data.map((d) => d[selectedMetric]);
    }
    return [];
  }, [metrics.data, selectedMetric]);

  const metricsOptions = useMemo(() => {
    if (metrics.data) {
      return Object.keys(metrics.data[0]).filter((x) => x !== 'id');
    }
    return [];
  }, [metrics.data, selectedModel]);

  useEffect(() => {
    if (metricsOptions.length > 0) {
      setSelectedMetric(metricsOptions[0]);
    }
  }, [metricsOptions]);

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Stack direction="row" spacing={1}>
          <CategoryFilter
            label="Model"
            value={selectedModel}
            onChange={handleModelChange}
            options={evaluatedModels}
          />
          <CategoryFilter
            label="Metric"
            disabled={!selectedModel || metrics.isFetching || metricsOptions.length === 0}
            value={selectedMetric}
            onChange={handleMetricChange}
            options={metricsOptions}
          />
          <CategoryFilter
            label="Color scale"
            value={selectedColorScale}
            onChange={setSelectedColorScale}
            options={['Picnic', 'Bluered', 'Jet', 'YlGnBu']}
          />
          <CategoryFilter
            label="Marker size"
            value={selectedMarkerSize}
            onChange={setSelectedMarkerSize}
            options={[2, 3, 4, 5, 6]}
          />
        </Stack>
      </Grid>
      <Grid item xs={12}>
        <Grid container spacing={2} sx={{ height: 500 }}>
          <Grid item xs={8}>
            <Box position="relative">
              {metrics.isFetching && <PlotLoader />}
              <Paper sx={{ p: 2 }}>
                <EmbeddingsPlot
                  onUnselect={handlePlotUnselect}
                  embeddings={embeddings.data}
                  onSelect={handlePlotSelect}
                  color={embeddingsColor}
                  resetIndex={plotResetIndex}
                  colorScale={selectedColorScale}
                  markerSize={selectedMarkerSize}
                  markerOpacity={0.5}
                  showScale
                />
              </Paper>
            </Box>
          </Grid>
          <Grid item xs={4} sx={{ height: '100%' }}>
            {selectedData.length > 0 && (
              <Paper sx={{ p: 2, height: '100%', overflow: 'auto' }}>
                {metricsType === 'user' ? (
                  <UsersDescription
                    attributes={itemAttributes}
                    split="validation"
                    users={selectedData}
                  />
                ) : (
                  <ItemsDescription attributes={itemAttributes} items={selectedData} />
                )}
              </Paper>
            )}
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
}

MetricsEmbeddings.defaultProps = {
  evaluatedModels: [],
};

MetricsEmbeddings.propTypes = {
  evaluatedModels: pt.arrayOf(pt.string),
  itemAttributes: pt.any.isRequired,
  metricsType: pt.string.isRequired,
};

export default MetricsEmbeddings;
