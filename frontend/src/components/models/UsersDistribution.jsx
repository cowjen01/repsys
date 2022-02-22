import React, { useState, useMemo, useRef } from 'react';
import pt from 'prop-types';
import { Paper, Grid, Box, Tabs, Tab, Stack } from '@mui/material';
import Plotly from 'plotly.js';

import { ScatterPlot, HistogramPlot } from '../plots';
import UsersDescription from '../dataset/UsersDescription';
import { useGetMetricsByModelQuery, useGetUsersEmbeddingsQuery } from '../../api';
import { CategoryFilter } from '../filters';
import TabPanel from '../TabPanel';
import { PlotLoader } from '../loaders';

function UsersDistribution({ metricsData }) {
  const models = Object.keys(metricsData.results);
  const metrics = metricsData.metrics.distributed.users;

  const [activeTab, setActiveTab] = useState(0);
  const [selectedData, setSelectedData] = useState();
  const [selectedModel, setSelectedModel] = useState(models[0]);
  const [selectedMetric, setSelectedMetric] = useState(metrics[0]);

  const histRef = useRef();

  const embeddings = useGetUsersEmbeddingsQuery('validation');
  const modelMetrics = useGetMetricsByModelQuery(selectedModel);

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
    setSelectedMetric(metrics[0]);
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
        users: points.map((p) => embeddings.data[p].id),
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
    if (modelMetrics.data) {
      return modelMetrics.data.map((d) => d[selectedMetric]);
    }
    return [];
  }, [modelMetrics.data, selectedMetric]);

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Stack direction="row" spacing={2}>
          <CategoryFilter
            label="Model"
            value={selectedModel}
            onChange={handleModelChange}
            options={models}
          />
          <CategoryFilter
            label="Metric"
            disabled={modelMetrics.isFetching}
            value={selectedMetric}
            onChange={handleMetricChange}
            options={metrics}
          />
        </Stack>
      </Grid>
      <Grid item xs={12}>
        <Grid container spacing={2} sx={{ height: 450 }}>
          <Grid item xs={8}>
            <Box position="relative">
              {modelMetrics.isFetching && <PlotLoader />}
              <Paper sx={{ p: 2 }}>
                <HistogramPlot
                  data={histogramData}
                  height={400}
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
                    height={350}
                    x={scatterPoints.x}
                    y={scatterPoints.y}
                    highlighted={selectedData.indices}
                    dragMode="pan"
                  />
                </TabPanel>
                <TabPanel
                  value={activeTab}
                  index={1}
                  sx={{ p: 2, overflow: 'auto', height: 'calc(100% - 48px)' }}
                >
                  <UsersDescription users={selectedData.users} />
                </TabPanel>
              </Paper>
            )}
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
}

UsersDistribution.propTypes = {
  // eslint-disable-next-line react/forbid-prop-types
  metricsData: pt.any.isRequired,
};

export default UsersDistribution;
