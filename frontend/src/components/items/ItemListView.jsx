import React from 'react';
import pt from 'prop-types';
import { ListItemAvatar, Avatar, ListItemText, ListItem } from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';

const typographyProps = {
  style: {
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
};

function ItemListView({ title, subtitle, image, style }) {
  return (
    <ListItem disableGutters style={style}>
      <ListItemAvatar>
        {image ? (
          <Avatar src={image} />
        ) : (
          <Avatar>
            <ImageIcon />
          </Avatar>
        )}
      </ListItemAvatar>
      <ListItemText
        primaryTypographyProps={typographyProps}
        secondaryTypographyProps={typographyProps}
        primary={title}
        secondary={subtitle}
      />
    </ListItem>
  );
}

ItemListView.defaultProps = {
  subtitle: '',
  image: null,
  style: {},
};

ItemListView.propTypes = {
  subtitle: pt.string,
  image: pt.string,
  title: pt.string.isRequired,
  // eslint-disable-next-line react/forbid-prop-types
  style: pt.any,
};

export default ItemListView;
