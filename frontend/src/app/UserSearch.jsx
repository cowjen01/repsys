import React, { useState, useEffect, useMemo } from 'react';
// import pt from 'prop-types';
import Box from '@mui/material/Box';
import Chart from 'react-google-charts';

const colors = ['#2196f3', '#03a9f4', '#4caf50', '#9c27b0', '#ff9800', '#cddc39', '#ffeb3b'];

function ModelMetrics() {
  const [scatterData, setScatterData] = useState([]);

  useEffect(() => {
    fetch('https://amp.pharm.mssm.edu/scavi/graph/GSE48968/tSNE/3')
      .then((response) => response.json())
      .then((d) => setScatterData(d));
  }, []);

  const preparedData = useMemo(
    () => scatterData.map((p) => [p.x, p.y, `fill-color: ${colors[p['KMeans-clustering'] % 7]}`]),
    [scatterData]
  );

  return (
    <Box
      sx={{
        p: 2,
      }}
    >
      <Chart
        width={400}
        height={400}
        chartType="ScatterChart"
        loader={<div>Loading Chart</div>}
        data={[['X', 'Y', { type: 'string', role: 'style' }], ...preparedData]}
        options={{
          title: 'Age vs. Weight comparison',
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
        // chartEvents={[
        //   {
        //     eventName: 'select',
        //     callback: ({ chartWrapper }) => {
        //       const chart = chartWrapper.getChart();
        //       const selection = chart.getSelection();
        //       if (selection.length === 1) {
        //         const [selectedItem] = selection;
        //         const dataTable = chartWrapper.getDataTable();
        //         const { row, column } = selectedItem;
        //         alert(
        //           'You selected : ' +
        //             JSON.stringify({
        //               row,
        //               column,
        //               value: dataTable.getValue(row, column),
        //             }),
        //           null,
        //           2
        //         );
        //       }
        //       console.log(selection);
        //     },
        //   },
        // ]}
        rootProps={{ 'data-testid': '1' }}
      />
    </Box>
  );
}

export default ModelMetrics;
