import React, { useMemo } from 'react';
// import pt from 'prop-types';
import Box from '@mui/material/Box';
import Chart from 'react-google-charts';

import { fetchItems } from './api';

const colors = ['#2196f3', '#03a9f4', '#4caf50', '#9c27b0', '#ff9800', '#cddc39', '#ffeb3b'];

function ModelMetrics({ onUserSelect }) {
  const { items, isLoading } = fetchItems('/userSpace');

  const preparedData = useMemo(
    () => items.map((p) => [p.x, p.y, `fill-color: ${colors[p.cluster]}`]),
    [items]
  );

  if (isLoading) {
    return null;
  }

  return (
    <Box
      sx={{
        p: 2,
      }}
    >
      <Chart
        width={400}
        height={350}
        chartType="ScatterChart"
        loader={<div>Loading Chart</div>}
        data={[['X', 'Y', { type: 'string', role: 'style' }], ...preparedData]}
        options={{
          chartArea: { left: 40, top: 20, bottom: 20, right: 20, width: '100%', height: '100%' },
          dataOpacity: 1,
          enableInteractivity: true,
          pointSize: 3,
          legend: 'none',
          // explorer: {
          //   keepInBounds: true,
          //   maxZoomIn: 3,
          //   maxZoomOut: 1,
          //   zoomDelta: 1.1
          // }
        }}
        chartEvents={[
          {
            eventName: 'select',
            callback: ({ chartWrapper }) => {
              const chart = chartWrapper.getChart();
              const selection = chart.getSelection();
              if (selection.length === 1) {
                const [selectedItem] = selection;
                const { row } = selectedItem;
                if (items[row]) {
                  onUserSelect(items[row].id);
                }
              }
            },
          },
        ]}
      />
    </Box>
  );
}

export default ModelMetrics;
