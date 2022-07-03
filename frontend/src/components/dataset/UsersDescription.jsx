import React, { useEffect } from 'react';
import pt from 'prop-types';
import { Typography, Stack, Box, List, Alert } from '@mui/material';

import { PanelLoader } from '../loaders';
import { ItemListView } from '../items';
import ErrorAlert from '../ErrorAlert';
import { useDescribeUsersMutation } from '../../api';
import AttributesPlot from './AttributesPlot';

function UsersDescription({ attributes, users, split }) {
  const [describeUsers, { data, error, isError, isLoading, isUninitialized }] =
    useDescribeUsersMutation();

  useEffect(() => {
    if (users.length) {
      describeUsers({ users, split });
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

  return (
    <Stack spacing={1}>
      <Box>
        <Typography variant="h6" sx={{ fontSize: '1rem' }}>
          Interacted Items
        </Typography>
        <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
          A list of the most interacted items
        </Typography>
        <List dense>
          {data.topItems.map((item) => (
            <ItemListView key={item.id} item={item} style={{ paddingLeft: 5 }} />
          ))}
        </List>
        <AttributesPlot attributes={attributes} description={data.itemsDescription} />
      </Box>
      <Alert severity="info">Disable selection by double-clicking inside the plot area.</Alert>
    </Stack>
  );
}

UsersDescription.defaultProps = {
  split: 'train',
};

UsersDescription.propTypes = {
  split: pt.string,
  attributes: pt.any.isRequired,
  users: pt.arrayOf(pt.number).isRequired,
};

export default UsersDescription;
