import React, { useState, useEffect } from 'react';
import pt from 'prop-types';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Pagination from '@mui/material/Pagination';
import Skeleton from '@mui/material/Skeleton';
import BarChartIcon from '@mui/icons-material/BarChart';
import Chip from '@mui/material/Chip';
import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';
import LabelIcon from '@mui/icons-material/Label';
import Stack from '@mui/material/Stack';
import LightbulbIcon from '@mui/icons-material/Lightbulb';
import BookmarkIcon from '@mui/icons-material/Bookmark';
import Box from '@mui/material/Box';
import IconButton from '@mui/material/IconButton';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import KeyboardArrowLeftIcon from '@mui/icons-material/KeyboardArrowLeft';

import ItemCardView from './ItemCardView';
import { postRequest } from './api';

function RecommenderView({
  title,
  model,
  selectedUser,
  customInteractions,
  itemsPerPage,
  itemsLimit,
  onItemClick,
  modelParams,
  onMetricsClick,
}) {
  const [page, setPage] = useState(0);
  const { items, isLoading } = postRequest('/predict', {
    model,
    ...(selectedUser
      ? {
          user: selectedUser.id,
        }
      : {
          interactions: customInteractions.map((x) => x.id),
        }),
    params: modelParams[model],
    limit: itemsLimit,
  });

  const handlePageChange = (e, newPage) => {
    setPage(newPage - 1);
  };

  useEffect(() => {
    setPage(0);
  }, [selectedUser]);

  return (
    <Grid container spacing={1}>
      <Grid item xs={12}>
        <Grid container alignItems="center" justifyContent="space-between">
          <Grid item>
            {/* <Stack direction="row" spacing={1}>
              <Chip
                // color="primary"
                size="small"
                // clickable
                // variant="outlined"
                icon={<BookmarkIcon />}
                label={title}
              />
              <Chip
                // color="primary"
                size="small"
                // clickable
                // variant="outlined"
                icon={<LightbulbIcon />}
                label={model}
              />
            </Stack> */}
            <Typography variant="h6" component="div">
              {title}
            </Typography>
          </Grid>
          <Grid item>
            <Stack direction="row">
              <IconButton disabled={page === 0} onClick={() => setPage(page - 1)}>
                <KeyboardArrowLeftIcon />
              </IconButton>
              <IconButton
                disabled={page === Math.ceil(items.length / itemsPerPage) - 1}
                onClick={() => setPage(page + 1)}
              >
                <KeyboardArrowRightIcon />
              </IconButton>
            </Stack>
          </Grid>
        </Grid>
      </Grid>

      <Grid item xs={12}>
        <Grid container spacing={2}>
          {!isLoading
            ? items.slice(itemsPerPage * page, itemsPerPage * (page + 1)).map((item) => (
                <Grid key={item.id} item xs={12} md={12 / itemsPerPage}>
                  <ItemCardView
                    title={item.title}
                    subtitle={item.subtitle}
                    caption={item.caption}
                    image={item.image}
                    imageHeight={Math.ceil(600 / itemsPerPage)}
                    onClick={() => onItemClick(item)}
                  />
                </Grid>
              ))
            : [...Array(itemsPerPage).keys()].map((i) => (
                <Grid key={i} item display="flex" md={12 / itemsPerPage}>
                  <Skeleton variant="rectangular" height={155} width="100%" />
                </Grid>
              ))}
          {!isLoading && items.length === 0 && (
            <Grid item xs={12}>
              <Alert severity="warning">
                <AlertTitle>No recommended items</AlertTitle>
                The model has not returned any recommendations.
              </Alert>
            </Grid>
          )}
        </Grid>
      </Grid>
    </Grid>
  );
}

RecommenderView.defaultProps = {
  modelParams: {},
  customInteractions: [],
};

RecommenderView.propTypes = {
  title: pt.string.isRequired,
  itemsPerPage: pt.number.isRequired,
  model: pt.string.isRequired,
  onMetricsClick: pt.func.isRequired,
  customInteractions: pt.array,
  // eslint-disable-next-line react/forbid-prop-types
  modelParams: pt.any,
};

export default RecommenderView;
