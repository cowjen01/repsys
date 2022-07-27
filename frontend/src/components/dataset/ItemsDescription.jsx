import React, { useEffect } from 'react';
import pt from 'prop-types';

import { Stack, Alert } from '@mui/material';
import { PanelLoader } from '../loaders';
import { useDescribeItemsMutation } from '../../api';
import ErrorAlert from '../ErrorAlert';
import AttributesPlot from './AttributesPlot';

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
    <Stack spacing={1}>
      <AttributesPlot attributes={attributes} description={data.description} />
      <Alert severity="info">Disable selection by double-clicking inside the plot area.</Alert>
    </Stack>
  );
}

ItemsDescription.propTypes = {
  items: pt.arrayOf(pt.number).isRequired,
  // eslint-disable-next-line react/forbid-prop-types
  attributes: pt.any.isRequired,
};

export default ItemsDescription;
