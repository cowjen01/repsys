import React from 'react';
import pt from 'prop-types';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';

function ItemCardView({ caption, title, subtitle, image, imageHeight }) {
  return (
    <Card sx={{ width: '100%' }}>
      {image && (
        <CardMedia
          sx={{ height: imageHeight, objectPosition: 'top' }}
          component="img"
          image={image}
        />
      )}
      <CardContent>
        {caption && (
          <Typography sx={{ fontSize: 14 }} color="text.secondary" gutterBottom>
            {caption}
          </Typography>
        )}
        <Typography sx={{ fontSize: 18 }} component="div" gutterBottom>
          {title}
        </Typography>
        {subtitle && <Typography color="text.secondary">{subtitle}</Typography>}
      </CardContent>
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
};

export default ItemCardView;
