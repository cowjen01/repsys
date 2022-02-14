import React, { useState, useRef, useMemo, useEffect } from 'react';
import pt from 'prop-types';
import { Paper, Stack, Box } from '@mui/material';
import Plotly from 'plotly.js';

import { CategoryFilter } from '../filters';
import { HistogramPlot } from '../plots';
import { useGetMetricsByModelQuery } from '../../api';

function MetricsHistogram({ onSelect, summaryData }) {
  const [isLoading, setIsLoading] = useState(false);
  const [histData, setHistData] = useState([]);
  const [selectedModel, setSelectedModel] = useState(Object.keys(summaryData)[0]);
  const [selectedMetric, setSelectedMetric] = useState('');

  const histRef = useRef();

  const metrics = useGetMetricsByModelQuery(selectedModel);

  // useEffect(() => {
  //   async function fetchData() {
  //     setIsLoading(true);
  //     await sleep(1000);
  //     setIsLoading(false);
  //   }
  //   fetchData();
  //   setHistData(histogramData.SVD);
  //   setSelectedModel('SVD');
  //   setSelectedMetric('Recall@20');
  // }, []);

  const resetHistSelection = () => {
    Plotly.restyle(histRef.current.el, { selectedpoints: [null] });
  };

  const resetSelection = () => {
    onSelect({
      ids: [],
      points: [],
    });
  };

  const handleFilterApply = async () => {
    // setIsLoading(true);
    // await sleep(500);
    // setIsLoading(false);
  };

  const modelMetricsOptions = useMemo(
    () =>
      Object.keys(summaryData[selectedModel].current).map((metric) => ({
        value: metric,
        label: metric,
      })),
    [selectedModel]
  );

  const histogramPoints = useMemo(() => {
    if (metrics.isSuccess) {
      return {
        x: metrics.data.map((d) => d[selectedMetric]),
        meta: metrics.data.map(({ id }) => ({ id })),
      };
    }
    return {
      x: [],
      meta: [],
    };
  }, [metrics.isLoading, selectedMetric]);

  return (
    <Paper sx={{ p: 2 }}>
      <Stack direction="row" spacing={2}>
        <CategoryFilter
          label="Model"
          value={selectedModel}
          onChange={(newValue) => {
            setSelectedModel(newValue);
            // setSelectedMetric(Object.keys(histData[selectedModel][0])[0]);
            resetHistSelection();
            resetSelection();
          }}
          options={Object.keys(summaryData).map((model) => ({
            value: model,
            label: model,
          }))}
        />
        <CategoryFilter
          label="Metric"
          value={selectedMetric}
          onBlur={handleFilterApply}
          onChange={(newValue) => {
            setSelectedMetric(newValue);
            resetHistSelection();
            resetSelection();
          }}
          options={modelMetricsOptions}
        />
      </Stack>
      <HistogramPlot
        data={histogramPoints.x}
        meta={histogramPoints.meta}
        height={400}
        innerRef={histRef}
        onDeselect={() => {
          resetSelection();
        }}
        onSelected={(eventData) => {
          if (eventData && eventData.points.length) {
            const { points } = eventData;
            onSelect({
              ids: points.map((p) => p.customdata.id),
              points: points[0].data.selectedpoints,
            });
          }
        }}
      />
    </Paper>
  );
}

MetricsHistogram.defaultProps = {};

MetricsHistogram.propTypes = {
  onSelect: pt.func.isRequired,
};

export default MetricsHistogram;
