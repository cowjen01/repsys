import React, { useEffect } from 'react';
import { Paper, Skeleton, Typography } from '@mui/material';
import { useSelector, useDispatch } from 'react-redux';
import { FixedSizeList } from 'react-window';

import { ItemListView } from '../items';
import { customInteractionsSelector, selectedUserSelector } from '../../reducers/root';
import {
  fetchInteractions,
  interactionsSelector,
  interactionsStatusSelector,
} from '../../reducers/interactions';

const INTERACTIONS_HEIGHT = 330;

function renderRow({ index, style, data }) {
  const item = data[index];
  return (
    <ItemListView
      style={style}
      image={item.image}
      key={item.id}
      id={item.id}
      title={item.title}
      subtitle={item.subtitle}
    />
  );
}

function UserInteractionsList() {
  const dispatch = useDispatch();
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);
  const userInteractions = useSelector(interactionsSelector);
  const status = useSelector(interactionsStatusSelector);

  useEffect(() => {
    if (selectedUser) {
      dispatch(fetchInteractions(selectedUser.id));
    }
  }, [selectedUser, dispatch]);

  const interactions = customInteractions.length ? customInteractions : userInteractions;

  if (selectedUser && status !== 'succeeded') {
    return <Skeleton variant="rectangular" height={INTERACTIONS_HEIGHT + 48} width="100%" />;
  }

  return (
    <Paper>
      <Typography
        sx={{
          paddingLeft: '16px',
          paddingRight: '16px',
          lineHeight: '48px',
          color: 'rgba(0, 0, 0, 0.6)',
        }}
        variant="subtitle2"
        component="div"
      >
        Interactions ({interactions.length})
      </Typography>
      <FixedSizeList
        height={INTERACTIONS_HEIGHT}
        itemData={interactions}
        itemSize={70}
        itemCount={interactions.length}
      >
        {renderRow}
      </FixedSizeList>
    </Paper>
  );
}

export default UserInteractionsList;
