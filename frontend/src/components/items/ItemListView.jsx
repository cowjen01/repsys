import React from 'react';
import pt from 'prop-types';
import { ListItemAvatar, Avatar, ListItemText, ListItem } from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';

function ItemListView({ title, subtitle, image }) {
  return (
    <ListItem>
      <ListItemAvatar>
        {image ? (
          <Avatar src={image} />
        ) : (
          <Avatar>
            <ImageIcon />
          </Avatar>
        )}
      </ListItemAvatar>
      <ListItemText primary={title} secondary={subtitle} />
    </ListItem>
  );
}

ItemListView.defaultProps = {
  subtitle: '',
  image: '',
};

ItemListView.propTypes = {
  subtitle: pt.string,
  image: pt.string,
  title: pt.string.isRequired,
};

export default ItemListView;
