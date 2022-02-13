import React, { useEffect, useState } from 'react';
import { Paper, Typography, Stack, Box, Chip } from '@mui/material';

import PanelLoader from '../PanelLoader';
import BarPlotHistogram from './BarPlotHistogram';
import { sleep, capitalize } from '../../utils';

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

function ItemDescriptionPanel({ columns, itemIds }) {
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    async function loadData() {
      setIsLoading(true);
      await sleep(500);
      setIsLoading(false);
    }

    if (itemIds.length) {
      loadData();
    }
  }, [itemIds]);

  if (!itemIds.length) {
    return null;
  }

  if (isLoading) {
    return <PanelLoader />;
  }

  return (
    <Paper sx={{ p: 2, maxHeight: '100%', overflow: 'auto' }}>
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
                  Attribute values distribution
                </Typography>
                <BarPlotHistogram bins={data.bins} hist={data.hist} />
              </>
            )}
          </Box>
        ))}
      </Stack>
    </Paper>
  );
}

export default ItemDescriptionPanel;
