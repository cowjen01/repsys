import React, { useState, useMemo, useRef, useEffect } from 'react';
import pt from 'prop-types';
import { Paper, Grid, Box, Tabs, Tab, Stack, Typography } from '@mui/material';
import Plotly from 'plotly.js';

import { ScatterPlot, HistogramPlot } from '../plots';
import UsersDescription from '../dataset/UsersDescription';
import ItemsDescription from '../dataset/ItemsDescription';
import {
  useGetUserMetricsByModelQuery,
  useGetUserEmbeddingsQuery,
  useGetItemMetricsByModelQuery,
  useGetItemEmbeddingsQuery,
} from '../../api';
import { CategoryFilter } from '../filters';
import TabPanel from '../TabPanel';
import { PlotLoader } from '../loaders';

function MetricsDistribution({ metricsType, itemAttributes, evaluatedModels }) {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedData, setSelectedData] = useState();
  const [selectedModel, setSelectedModel] = useState(evaluatedModels[0]);
  const [selectedMetric, setSelectedMetric] = useState('');

  const histRef = useRef();

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

  const resetHistSelection = () => {
    Plotly.restyle(histRef.current.el, { selectedpoints: [null] });
  };

  const handleTabChange = (event, index) => {
    setActiveTab(index);
  };

  const resetSelection = () => {
    setSelectedData(undefined);
    resetHistSelection();
  };

  const handleModelChange = (newValue) => {
    setSelectedModel(newValue);
    // setSelectedMetric('');
    resetSelection();
  };

  const handleMetricChange = (newValue) => {
    setSelectedMetric(newValue);
    resetSelection();
  };

  const handleHistSelect = (eventData) => {
    if (eventData && eventData.points.length) {
      const points = eventData.points[0].data.selectedpoints;
      setSelectedData({
        indices: points,
        ids: points.map((p) => metrics.data[p].id),
      });
    }
  };

  const handleHistUnselect = () => {
    resetSelection();
  };

  const scatterPoints = useMemo(() => {
    if (embeddings.data) {
      return {
        x: embeddings.data.map(({ x }) => x),
        y: embeddings.data.map(({ y }) => y),
      };
    }
    return { x: [], y: [] };
  }, [embeddings.data]);

  const histogramData = useMemo(() => {
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
    if (!selectedMetric && metricsOptions.length > 0) {
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
        </Stack>
      </Grid>
      <Grid item xs={12}>
        <Grid container spacing={2} sx={{ height: 500 }}>
          <Grid item xs={8}>
            <Box position="relative">
              {metrics.isFetching && <PlotLoader />}
              <Paper sx={{ p: 2 }}>
                <HistogramPlot
                  data={histogramData}
                  innerRef={histRef}
                  onDeselect={handleHistUnselect}
                  onSelected={handleHistSelect}
                />
              </Paper>
            </Box>
          </Grid>
          <Grid item xs={4} sx={{ height: '100%' }}>
            {selectedData && (
              <Paper sx={{ height: '100%' }}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                  <Tabs value={activeTab} onChange={handleTabChange} variant="fullWidth">
                    <Tab label="Embeddings" />
                    <Tab label="Description" />
                  </Tabs>
                </Box>
                <TabPanel value={activeTab} index={0} sx={{ p: 1 }}>
                  <ScatterPlot
                    height={400}
                    x={scatterPoints.x}
                    y={scatterPoints.y}
                    highlighted={selectedData.indices}
                    markerSize={2}
                    markerOpacity={0.5}
                    dragMode="pan"
                  />
                </TabPanel>
                <TabPanel
                  value={activeTab}
                  index={1}
                  sx={{ p: 2, overflow: 'auto', height: 'calc(100% - 48px)' }}
                >
                  {metricsType === 'user' ? (
                    <UsersDescription
                      attributes={itemAttributes}
                      split="validation"
                      users={selectedData.ids}
                    />
                  ) : (
                    <ItemsDescription attributes={itemAttributes} items={selectedData.ids} />
                  )}
                </TabPanel>
              </Paper>
            )}
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
}

MetricsDistribution.defaultProps = {
  evaluatedModels: [],
};

MetricsDistribution.propTypes = {
  evaluatedModels: pt.arrayOf(pt.string),
  itemAttributes: pt.any.isRequired,
  metricsType: pt.string.isRequired,
};

export default MetricsDistribution;
