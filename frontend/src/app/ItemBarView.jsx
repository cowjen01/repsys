import React, { useState } from 'react';
import pt from 'prop-types';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Pagination from '@mui/material/Pagination';

import ItemView from './ItemView';
import { fetchItems } from './api';
import ItemSkeleton from './ItemSkeleton';

function ItemBarView({ title, model, itemsPerPage, modelAttributes }) {
  const [page, setPage] = useState(0);
  const { items, isLoading } = fetchItems(
    `/recommendations?model=${model}&attributes=${encodeURIComponent(
      JSON.stringify(modelAttributes[model])
    )}`
  );

  const handlePageChange = (e, newPage) => {
    setPage(newPage - 1);
  };

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Typography variant="h6" component="div">
          {title}
        </Typography>
      </Grid>
      <Grid item xs={12}>
        <Grid container spacing={2} alignItems="stretch">
          {items.slice(itemsPerPage * page, itemsPerPage * (page + 1)).map((item) => (
            <Grid key={item.id} item display="flex" md={12 / itemsPerPage}>
              <ItemView
                title={item.title}
                subtitle={item.subtitle.toString()}
                header={item.header.toString()}
                description={item.description}
                image={item.image}
              />
            </Grid>
          ))}
        </Grid>
      </Grid>
      {items.length > itemsPerPage && (
        <Grid item xs={12}>
          <Pagination
            page={page + 1}
            onChange={handlePageChange}
            count={Math.ceil(items.length / itemsPerPage)}
          />
        </Grid>
      )}
    </Grid>
  );
}

ItemBarView.defaultProps = {
  modelAttributes: {},
};

ItemBarView.propTypes = {
  title: pt.string.isRequired,
  itemsPerPage: pt.number.isRequired,
  model: pt.string.isRequired,
  // eslint-disable-next-line react/forbid-prop-types
  modelAttributes: pt.any,
};

export default ItemBarView;
