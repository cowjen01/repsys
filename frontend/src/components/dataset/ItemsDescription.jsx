import React, { useEffect } from 'react';
import pt from 'prop-types';
import { Typography, Stack, Box, Chip } from '@mui/material';

import PanelLoader from '../PanelLoader';
import BarPlotHistogram from './BarPlotHistogram';
import { capitalize } from '../../utils';
import { useDescribeItemsMutation } from '../../api';
import ErrorAlert from '../ErrorAlert';

function ItemsDescription({ attributes, items }) {
  const [describeItems, { data, error, isError, isLoading, isUninitialized }] =
    useDescribeItemsMutation();

  useEffect(() => {
    if (items.length) {
      describeItems({ items });
    }
  }, [items]);

  if (isUninitialized) {
    return null;
  }

  if (isError) {
    return <ErrorAlert error={error} />;
  }

  if (isLoading) {
    return <PanelLoader />;
  }

  return (
    <Stack spacing={2}>
      {Object.entries(data.attributes).map(([key, { topValues, bins, hist }]) => (
        <Box key={key}>
          <Typography variant="h6" sx={{ fontSize: '1rem' }}>
            {capitalize(key)}
          </Typography>
          {(attributes[key].dtype === 'tags' || attributes[key].dtype === 'category') && (
            <>
              <Typography gutterBottom variant="body2">
                The most frequent values
              </Typography>
              <Stack direction="row" spacing={1}>
                {topValues.map((value) => (
                  <Chip key={value} label={value} />
                ))}
              </Stack>
            </>
          )}
          {attributes[key].dtype === 'number' && (
            <>
              <Typography gutterBottom variant="body2">
                Attribute values distribution
              </Typography>
              <BarPlotHistogram bins={bins} hist={hist} />
            </>
          )}
        </Box>
      ))}
    </Stack>
  );
}

ItemsDescription.propTypes = {
  items: pt.arrayOf(pt.number).isRequired,
  // eslint-disable-next-line react/forbid-prop-types
  attributes: pt.any.isRequired,
};

export default ItemsDescription;
