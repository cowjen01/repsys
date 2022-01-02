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
import { itemFieldsSelector } from '../../reducers/settings';

const INTERACTIONS_HEIGHT = 360;

function renderRow({ index, style, data }) {
  const { interactions, itemFields } = data;
  const item = interactions[index];
  return (
    <ItemListView
      style={style}
      key={item.id}
      id={item.id}
      title={item[itemFields.title]}
      subtitle={item[itemFields.subtitle]}
      image={item[itemFields.image]}
    />
  );
}

function InteractionsList() {
  const dispatch = useDispatch();
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);
  const userInteractions = useSelector(interactionsSelector);
  const status = useSelector(interactionsStatusSelector);
  const itemFields = useSelector(itemFieldsSelector);

  useEffect(() => {
    if (selectedUser) {
      dispatch(fetchInteractions(selectedUser));
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
          lineHeight: '48px',
          color: 'text.secondary',
          pl: '16px',
        }}
        variant="subtitle2"
        component="div"
      >
        Interactions ({interactions.length})
      </Typography>
      <FixedSizeList
        height={INTERACTIONS_HEIGHT}
        itemData={{
          interactions,
          itemFields,
        }}
        itemSize={60}
        itemCount={interactions.length}
      >
        {renderRow}
      </FixedSizeList>
    </Paper>
  );
}

export default InteractionsList;
