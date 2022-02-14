import React, { useMemo, useState } from 'react';
import pt from 'prop-types';
import { Grid, Paper, Box, Tabs, Tab } from '@mui/material';

import { IndicatorPlot, BarPlot } from '../plots';

function MetricsSummary({ metricsData }) {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event, index) => {
    setActiveTab(index);
  };

  const models = Object.keys(metricsData.results);
  const currentModel = models[activeTab];

  const indicators = useMemo(() => {
    const results = metricsData.results[currentModel];
    return Object.entries(results.current).map(([metric, value]) => ({
      name: metric,
      value,
      delta: (results.previous && results.previous[metric]) || 0,
    }));
  }, [activeTab]);

  const barPlotData = useMemo(
    () =>
      Object.entries(metricsData.results).map(([model, results]) => ({
        y: Object.keys(results.current),
        x: Object.values(results.current),
        name: model,
      })),
    []
  );

  return (
    <Grid container spacing={2}>
      <Grid item xs={7}>
        <Paper sx={{ height: '100%' }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={activeTab} onChange={handleTabChange} variant="fullWidth">
              {models.map((model) => (
                <Tab label={model} key={model} />
              ))}
            </Tabs>
          </Box>
          <Box sx={{ p: 2 }}>
            <Grid container>
              {indicators.map((indicator) => (
                <Grid item xs={3} key={indicator.name}>
                  <IndicatorPlot
                    height={150}
                    title={indicator.name}
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
  );
}

MetricsSummary.propTypes = {
  // eslint-disable-next-line react/forbid-prop-types
  metricsData: pt.any.isRequired,
};

export default MetricsSummary;
