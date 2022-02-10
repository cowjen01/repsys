import React, { useEffect, useState } from 'react';
import { Paper, Typography, Stack, Box, Alert, Chip, CircularProgress } from '@mui/material';

import { BarPlot } from '../plots';
import PanelLoader from '../PanelLoader';

const characteristics = {
  genres: {
    topValues: ['action', 'drama', 'horror', 'comedy'],
  },
  country: {
    topValues: ['CO', 'MK', 'FR', 'CZ', 'SK', 'RU'],
  },
  year: {
    hist: [2, 4, 2, 10, 12],
    bins: [0, 2000, 2006, 2020, 2050, 2070],
  },
};

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function ItemDescriptionPanel({ columns, selectedItems }) {
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    async function loadData() {
      setIsLoading(true);
      await sleep(500);
      setIsLoading(false);
    }

    if (selectedItems.length) {
      loadData();
    }
  }, [selectedItems]);

  if (!selectedItems.length) {
    return null;
  }

  if (isLoading) {
    return <PanelLoader />;
  }

  return (
    <Paper sx={{ p: 2 }}>
      <Stack spacing={2}>
        {Object.entries(characteristics).map(([col, data]) => (
          <Box key={col}>
            <Typography variant="h6" sx={{ fontSize: '1.1rem' }}>
              {capitalize(col)}
            </Typography>
            {(columns[col].dtype === 'tags' || columns[col].dtype === 'category') && (
              <>
                <Typography gutterBottom variant="body2">
                  The most frequent values
                </Typography>
                <Stack direction="row" spacing={1}>
                  {data.topValues.map((value) => (
                    <Chip key={value} label={value} />
                  ))}
                </Stack>
              </>
            )}
            {columns[col].dtype === 'number' && (
              <>
                <Typography gutterBottom variant="body2">
                  Values distribution
                </Typography>
                <BarPlot
                  height={150}
                  layoutProps={{
                    bargap: 0,
                    xaxis: {
                      tickfont: { size: 10 },
                    },
                  }}
                  data={[
                    {
                      x: data.hist.map((_, index) => `${data.bins[index]}-${data.bins[index + 1]}`),
                      y: data.hist,
                    },
                  ]}
                />
              </>
            )}
          </Box>
        ))}
      </Stack>
    </Paper>
  );
}

export default ItemDescriptionPanel;
