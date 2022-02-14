import React, { useEffect } from 'react';
import pt from 'prop-types';
import { Typography, Stack, Box, List } from '@mui/material';

import BarPlotHistogram from './BarPlotHistogram';
import PanelLoader from '../PanelLoader';
import { ItemListView } from '../items';
import ErrorAlert from '../ErrorAlert';
import { useDescribeUsersMutation } from '../../api';

function UsersDescription({ users }) {
  const [describeUsers, { data, error, isError, isLoading, isUninitialized }] =
    useDescribeUsersMutation();

  useEffect(() => {
    if (users.length) {
      describeUsers({ users });
    }
  }, [users]);

  if (isUninitialized) {
    return null;
  }

  if (isError) {
    return <ErrorAlert error={error} />;
  }

  if (isLoading) {
    return <PanelLoader />;
  }

  const { distribution, topItems } = data.interactions;

  return (
    <Stack spacing={2}>
      <Box>
        <Typography variant="h6" sx={{ fontSize: '1rem' }}>
          Interacted Items
        </Typography>
        <Typography gutterBottom variant="body2">
          A list of the most interacted items
        </Typography>
        <List dense>
          {topItems.map((item) => (
            <ItemListView key={item.id} item={item} style={{ paddingLeft: 5 }} />
          ))}
        </List>
      </Box>
      <Box>
        <Typography variant="h6" sx={{ fontSize: '1rem' }}>
          Interactions Distribution
        </Typography>
        <Typography gutterBottom variant="body2">
          A distribution of the total number of interactions
        </Typography>
        <BarPlotHistogram bins={distribution.bins} hist={distribution.hist} />
      </Box>
    </Stack>
  );
}

UsersDescription.propTypes = {
  users: pt.arrayOf(pt.number).isRequired,
};

export default UsersDescription;
