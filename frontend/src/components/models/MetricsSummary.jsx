import React, { useMemo, useState } from 'react';
import { Grid, Paper, Box, Tabs, Tab, Typography } from '@mui/material';

import { IndicatorPlot, BarPlot } from '../plots';

function MetricsSummary({ summaryData }) {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event, index) => {
    setActiveTab(index);
  };

  const indicators = useMemo(() => {
    const model = Object.keys(summaryData)[activeTab];
    return Object.entries(summaryData[model].current).map((metric) => {
      const prevResults = summaryData[model].previous;
      return {
        name: metric[0],
        value: metric[1],
        delta: prevResults && prevResults[metric[0]] ? prevResults[metric[0]] : 0,
      };
    });
  }, [activeTab]);

  const barPlotData = useMemo(
    () =>
      Object.entries(summaryData).map((model) => ({
        y: Object.keys(model[1].current),
        x: Object.values(model[1].current),
        name: model[0],
      })),
    []
  );

  return (
    <>
      <Box pl={1}>
        <Typography component="div" variant="h6">
          Models Performance
        </Typography>
        <Typography variant="subtitle1" gutterBottom>
          A performance in the individual metrics with comparasion to the previous evaluation
        </Typography>
      </Box>
      <Grid container spacing={2}>
        <Grid item xs={7}>
          <Paper sx={{ height: '100%' }}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={activeTab} onChange={handleTabChange} variant="fullWidth">
                {Object.keys(summaryData).map((model) => (
                  <Tab label={model} key={model} />
                ))}
              </Tabs>
            </Box>
            <Box sx={{ p: 2 }}>
              <Grid container>
                {indicators.map((indicator) => (
                  <Grid item xs={3} key={indicator.name}>
                    <IndicatorPlot
                      title={indicator.name}
                      height={150}
                      value={indicator.value}
                      delta={indicator.delta}
                    />
                  </Grid>
                ))}
              </Grid>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={5}>
          <Paper>
            <BarPlot
              orientation="h"
              data={barPlotData}
              layoutProps={{
                margin: { t: 30, b: 40, l: 120, r: 40 },
              }}
              height={400}
            />
          </Paper>
        </Grid>
      </Grid>
    </>
  );
}

export default MetricsSummary;
