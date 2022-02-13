import React, { useState } from 'react';
import pt from 'prop-types';
import { Typography, Card, CardContent, CardMedia, CardActionArea, Skeleton } from '@mui/material';
import { useSelector } from 'react-redux';

import { itemViewSelector } from '../../reducers/settings';

function ItemCardView({ item, imageHeight, onClick }) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const itemView = useSelector(itemViewSelector);

  return (
    <Card sx={{ width: '100%', height: '100%' }}>
      <CardActionArea
        sx={{ height: '100%', flexDirection: 'column', alignItems: 'stretch' }}
        onClick={onClick}
      >
        {item[itemView.image] && (
          <CardMedia
            sx={{
              height: imageHeight,
              objectPosition: 'top',
              display: !imageLoaded ? 'none' : 'block',
            }}
            component="img"
            image={item[itemView.image]}
            onLoad={() => setImageLoaded(true)}
          />
        )}
        {item[itemView.image] && !imageLoaded && (
          <Skeleton variant="rectangular" height={imageHeight} width="100%" />
        )}
        <CardContent>
          {item[itemView.caption] && (
            <Typography noWrap sx={{ fontSize: 13 }} color="text.secondary" gutterBottom>
              {item[itemView.caption]}
            </Typography>
          )}
          <Typography noWrap sx={{ fontSize: 16 }} component="div">
            {item[itemView.title]}
          </Typography>
          {item[itemView.subtitle] && (
            <Typography noWrap sx={{ fontSize: 15 }} color="text.secondary">
              {item[itemView.subtitle]}
            </Typography>
          )}
        </CardContent>
      </CardActionArea>
    </Card>
  );
}

ItemCardView.propTypes = {
  imageHeight: pt.number.isRequired,
  // eslint-disable-next-line react/forbid-prop-types
  item: pt.any.isRequired,
  onClick: pt.func.isRequired,
};

export default ItemCardView;
