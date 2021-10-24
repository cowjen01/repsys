import React, { useState } from 'react';
import pt from 'prop-types';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Pagination from '@mui/material/Pagination';
import Skeleton from '@mui/material/Skeleton';
import BarChartIcon from '@mui/icons-material/BarChart';
import Chip from '@mui/material/Chip';

import ItemView from './ItemView';
import { fetchItems } from './api';

function ItemBarView({
  title,
  model,
  user,
  customInteractions,
  itemsPerPage,
  modelAttributes,
  onMetricsClick,
}) {
  const [page, setPage] = useState(0);
  const { items, isLoading } = fetchItems('/recommendations', {
    model,
    ...(user
      ? {
          user,
        }
      : {
          interactions: JSON.stringify(customInteractions.map((x) => x.id)),
        }),
    attributes: JSON.stringify(modelAttributes[model]),
  });

  const handlePageChange = (e, newPage) => {
    setPage(newPage - 1);
  };

  return (
    <Grid container spacing={1}>
      <Grid item xs={12}>
        <Grid container spacing={2} alignItems="center">
          <Grid item>
            <Typography variant="h6" component="div">
              {title}
            </Typography>
          </Grid>
          <Grid item>
            <Chip
              color="secondary"
              onClick={onMetricsClick}
              size="small"
              clickable
              // variant="outlined"
              icon={<BarChartIcon />}
              label="Metrics"
            />
          </Grid>
        </Grid>
      </Grid>
      <Grid item xs={12}>
        <Grid container spacing={2} alignItems="stretch">
          {!isLoading
            ? items.slice(itemsPerPage * page, itemsPerPage * (page + 1)).map((item) => (
                <Grid key={item.id} item display="flex" xs={12} md={12 / itemsPerPage}>
                  <ItemView
                    id={item.id}
                    title={item.title}
                    subtitle={item.subtitle.toString()}
                    header={item.header.toString()}
                    description={item.description}
                    // image={item.image}
                    image="foo"
                    imageWidth={Math.ceil(1000 / itemsPerPage)}
                    imageHeight={Math.ceil(400 / itemsPerPage)}
                  />
                </Grid>
              ))
            : [...Array(itemsPerPage).keys()].map((i) => (
                <Grid key={i} item display="flex" md={12 / itemsPerPage}>
                  <Skeleton variant="rectangular" height={155} width="100%" />
                </Grid>
              ))}
        </Grid>
      </Grid>
      <Grid
        item
        xs={12}
        sx={{
          marginTop: 1,
        }}
      >
        {isLoading && <Skeleton variant="rectangular" height={32} width="30%" />}
        {!isLoading && items.length > itemsPerPage && (
          <Pagination
            page={page + 1}
            onChange={handlePageChange}
            count={Math.ceil(items.length / itemsPerPage)}
          />
        )}
      </Grid>
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
  onMetricsClick: pt.func.isRequired,
  // userId: pt.string.isRequired,
  // eslint-disable-next-line react/forbid-prop-types
  modelAttributes: pt.any,
};

export default ItemBarView;
