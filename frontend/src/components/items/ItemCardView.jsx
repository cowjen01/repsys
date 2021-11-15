import React from 'react';
import pt from 'prop-types';
import { Typography, Card, CardContent, CardMedia, CardActionArea } from '@mui/material';

function ItemCardView({ caption, title, subtitle, image, imageHeight, onClick }) {
  return (
    <Card sx={{ width: '100%', height: '100%' }}>
      <CardActionArea
        sx={{ height: '100%', flexDirection: 'column', alignItems: 'stretch' }}
        onClick={onClick}
      >
        {image && (
          <CardMedia
            sx={{ maxHeight: imageHeight, objectPosition: 'top' }}
            component="img"
            image={image}
          />
        )}
        <CardContent>
          {caption && (
            <Typography noWrap sx={{ fontSize: 13 }} color="text.secondary" gutterBottom>
              {caption}
            </Typography>
          )}
          <Typography noWrap sx={{ fontSize: 16 }} component="div">
            {title}
          </Typography>
          {subtitle && (
            <Typography noWrap sx={{ fontSize: 15 }} color="text.secondary">
              {subtitle}
            </Typography>
          )}
        </CardContent>
      </CardActionArea>
    </Card>
  );
}

ItemCardView.defaultProps = {
  caption: '',
  subtitle: '',
  image: '',
};

ItemCardView.propTypes = {
  imageHeight: pt.number.isRequired,
  caption: pt.string,
  subtitle: pt.string,
  image: pt.string,
  title: pt.string.isRequired,
  onClick: pt.func.isRequired,
};

export default ItemCardView;
