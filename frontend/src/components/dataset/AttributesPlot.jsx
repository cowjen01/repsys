import React from 'react';
import pt from 'prop-types';
import { Typography, Box, Stack } from '@mui/material';

import { capitalize } from '../../utils';
import { BarPlot } from '../plots';

function AttributesPlot({ description, attributes }) {
  return (
    <Stack spacing={1}>
      {Object.entries(description).map(([key, { labels, bins, values }]) => (
        <Box key={key}>
          <Typography variant="h6" sx={{ fontSize: '1rem' }}>
            {capitalize(key)}
          </Typography>
          {(attributes[key].dtype === 'tag' || attributes[key].dtype === 'category') && (
            <>
              <Typography variant="body2" sx={{ fontSize: '0.8rem' }} mb={1}>
                The most frequent values
              </Typography>
              <BarPlot
                height={150}
                data={[
                  {
                    x: labels,
                    y: values,
                  },
                ]}
              />
            </>
          )}
          {attributes[key].dtype === 'number' && (
            <>
              <Typography sx={{ fontSize: '0.8rem' }} variant="body2">
                Attribute&#39;s values distribution
              </Typography>
              <BarPlot
                height={150}
                data={[
                  {
                    x: values.map((_, index) => `${bins[index]}-${bins[index + 1]}`),
                    y: values,
                  },
                ]}
              />
            </>
          )}
        </Box>
      ))}
    </Stack>
  );
}

AttributesPlot.propTypes = {
  attributes: pt.any.isRequired,
  description: pt.any.isRequired,
};

export default AttributesPlot;
