import React from 'react';
import pt from 'prop-types';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';

function ItemView({ id, header, title, description, subtitle, image, imageWidth, imageHeight }) {
  return (
    <Card sx={{ width: '100%' }}>
      {image && (
        <CardMedia
          sx={{ minHeight: imageHeight + 50 }}
          component="img"
          image={`https://picsum.photos/seed/${id}/${imageWidth}/${imageHeight}`}
          alt="item media"
        />
      )}
      <CardContent>
        {header && (
          <Typography sx={{ fontSize: 14 }} color="text.secondary" gutterBottom>
            {header}
          </Typography>
        )}
        <Typography sx={{ fontSize: 18 }} component="div" gutterBottom>
          {title}
        </Typography>
        {subtitle && <Typography color="text.secondary">{subtitle}</Typography>}
        {description && <Typography variant="body2">{description}</Typography>}
      </CardContent>
    </Card>
  );
}

ItemView.defaultProps = {
  header: '',
  subtitle: '',
  description: '',
  image: '',
};

ItemView.propTypes = {
  id: pt.number.isRequired,
  imageWidth: pt.number.isRequired,
  imageHeight: pt.number.isRequired,
  header: pt.string,
  subtitle: pt.string,
  description: pt.string,
  image: pt.string,
  title: pt.string.isRequired,
};

export default ItemView;
