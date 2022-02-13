import React from 'react';
import pt from 'prop-types';
import { ListItemAvatar, Avatar, ListItemText, ListItem } from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';
import { useSelector } from 'react-redux';

import { itemViewSelector } from '../../reducers/settings';

const typographyProps = {
  style: {
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
};

function ItemListView({ style, item }) {
  const itemView = useSelector(itemViewSelector);

  return (
    <ListItem style={style}>
      <ListItemAvatar>
        {item[itemView.image] ? (
          <Avatar src={item[itemView.image]} />
        ) : (
          <Avatar>
            <ImageIcon />
          </Avatar>
        )}
      </ListItemAvatar>
      <ListItemText
        primaryTypographyProps={typographyProps}
        secondaryTypographyProps={typographyProps}
        primary={item[itemView.title]}
        secondary={item[itemView.subtitle]}
      />
    </ListItem>
  );
}

ItemListView.defaultProps = {
  style: {},
};

ItemListView.propTypes = {
  // eslint-disable-next-line react/forbid-prop-types
  item: pt.any.isRequired,
  // eslint-disable-next-line react/forbid-prop-types
  style: pt.any,
};

export default ItemListView;
