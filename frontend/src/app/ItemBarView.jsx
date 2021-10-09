import React, { useEffect, useState } from 'react';
import pt from 'prop-types';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Pagination from '@mui/material/Pagination';

import ItemView from './ItemView';
import ItemSkeleton from './ItemSkeleton';

function ItemBarView({ title, itemsPerPage, totalItems }) {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setTimeout(() => {
      setIsLoading(false);
    }, 1000);
  }, []);

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Typography variant="h6">{title}</Typography>
      </Grid>
      <Grid item xs={12}>
        <Grid container spacing={2}>
          {[...Array(itemsPerPage).keys()].map((i) => (
            <Grid key={i} item md={12 / itemsPerPage}>
              {isLoading ? <ItemSkeleton /> : <ItemView />}
            </Grid>
          ))}
        </Grid>
      </Grid>
      <Grid item xs={12}>
        <Pagination count={Math.round(totalItems / itemsPerPage)} />
      </Grid>
    </Grid>
  );
}

ItemBarView.defaultProps = {
  itemsPerPage: 4,
};

ItemBarView.propTypes = {
  title: pt.string.isRequired,
  itemsPerPage: pt.number,
  totalItems: pt.number.isRequired,
};

export default ItemBarView;
