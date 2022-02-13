import React, { useState } from 'react';
import pt from 'prop-types';
import { ListItemAvatar, Avatar, ListItemText, ListItem, Skeleton } from '@mui/material';
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
  const [imageLoaded, setImageLoaded] = useState(false);
  const itemView = useSelector(itemViewSelector);

  return (
    <ListItem style={style}>
      <ListItemAvatar>
        {item[itemView.image] && !imageLoaded && (
          <Skeleton variant="circular" width={40} height={40} />
        )}
        {item[itemView.image] && (
          <Avatar
            onLoad={() => setImageLoaded(true)}
            sx={{ display: !imageLoaded ? 'none' : 'block' }}
            src={item[itemView.image]}
          />
        )}
        {!item[itemView.image] && (
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
